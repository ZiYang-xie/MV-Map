import os
import time
import functools
import numpy as np
import torch.multiprocessing as multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo
from tqdm import tqdm

from dataloader import grid
from torch.utils.cpp_extension import load
from utils.misc import eval_sh, time_once

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['../cuda/render_utils.cpp', '../cuda/render_utils_kernel.cu']],
        verbose=True)


'''
Model
Adapted from DVGO: https://github.com/sunset1995/DirectVoxGO
'''
class VoxelNeRF(torch.nn.Module):
    def __init__(self, cfg, data_list):
        super(VoxelNeRF, self).__init__()
        # init representation
        self.rgbnet_config = cfg.rgbnet_config
        self.voxel_config = cfg.voxel_config
        self.voxel_size = cfg.voxel_size
        self.final_voxel_size = cfg.final_voxel_size
        self.voxel_size_ratio = self.voxel_size / self.final_voxel_size
        
        self.sh_base = cfg.model_config.sh_base
        if self.sh_base > 0:
            assert cfg.rgbnet_config == None
            self.voxel_config['feature_dim'] = (self.sh_base+1)**2 * 3
        self.viewbase_pe = cfg.model_config.viewbase_pe
        self.alpha_init = cfg.model_config.alpha_init
        self.fast_color_thres = cfg.model_config.fast_color_thres
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-self.alpha_init) - 1)]))
        #self.init_voxels(cfg, data_list)
        self.voxels = dict(self._get_voxels(data_list, range(len(data_list)), {}))
        for key in self.voxels.keys():
            self.voxels[key]['density'].cuda()
            self.voxels[key]['feature'].cuda()
            self.voxels[key]['mask_cache'].cuda()

        self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(self.viewbase_pe)]))

        # Shared RGB Net
        self.rgbnet = None
        if cfg.rgbnet_config:
            dim0 = self.rgbnet_config.rgbnet_dim
            if cfg.rgbnet_config.view_denpendent:
                dim0 += (3+3*self.viewbase_pe*2)
            self.rgbnet_direct = cfg.rgbnet_config.rgbnet_direct
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, self.rgbnet_config.rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(self.rgbnet_config.rgbnet_width, self.rgbnet_config.rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(self.rgbnet_config.rgbnet_depth-2)
                ],
                nn.Linear(self.rgbnet_config.rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)

    
    def init_voxels(self, cfg, data_list):
        # Load from file
        if cfg.ckpt_path is not None:
            self.voxels = {}
            return 
        
        jobs = []
        manager = multiprocessing.Manager()
        ret_voxels_dict = manager.dict()
        print("Init Voxels")
        data_len = len(data_list)
        split = np.array_split(range(data_len), 16)
        for i in range(16):
            p = multiprocessing.Process(target=self._get_voxels, args=(data_list, split[i], ret_voxels_dict))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
    
        self.voxels = dict(ret_voxels_dict)
        for key in self.voxels.keys():
            self.voxels[key]['density'].cuda()
            self.voxels[key]['feature'].cuda()
            self.voxels[key]['mask_cache'].cuda()

    def _get_voxels(self, data_list, split, ret_voxels_dict):
        for i in tqdm(split):
            data_dict = data_list[i]
            voxel_data = self._build_voxel(data_dict)
            ret_voxels_dict[data_dict['scene_name']] = voxel_data
        return ret_voxels_dict

    '''
      Build Voxel Grids
    '''
    def _build_voxel(self, 
                    data_dict):
        xyz_min, xyz_max = data_dict['xyz_min'], data_dict['xyz_max']
        world_size = data_dict['world_size']
        poses = torch.tensor(data_dict['poses'])
        density = grid.create_grid(
                self.voxel_config['grid_type'],            # 'DenseGrid' or 'TensoRFGrid'
                channels=1, world_size=world_size,
                xyz_min=xyz_min, xyz_max=xyz_max,
                config=None)
        density = self.maskout_near_cam_vox(density, xyz_min, xyz_max, world_size, poses[:,:3,3], 1.5)

        feature = grid.create_grid(
                self.voxel_config['grid_type'], 
                channels=self.voxel_config['feature_dim'], world_size=world_size,
                xyz_min=xyz_min, xyz_max=xyz_max,
                config=None)
        
        mask_cache_world_size = world_size
        mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=xyz_min, xyz_max=xyz_max)

        voxel_grid = dict(
            density=density,
            feature=feature,
            mask_cache=mask_cache,
        )
        return voxel_grid

    @torch.no_grad()
    def update_depth(self, scene_name, depth, zero_mask):
        depth = depth.detach()
        update_mask = depth[zero_mask] < self.voxels[scene_name]['updated_depth'][zero_mask]
        self.voxels[scene_name]['updated_depth'][zero_mask][update_mask] = depth[zero_mask][update_mask]
        return update_mask

    def create_img_feat_grid(self, feats, scene_name, C, xyz_min, xyz_max, world_size):
        img_feat_grid = grid.create_grid(
                            self.voxel_config['grid_type'], 
                            channels=C, world_size=world_size,
                            xyz_min=xyz_min, xyz_max=xyz_max,
                            config=None)
        X,Y,Z = feats.shape[-3:]
        img_feat_grid.grid = torch.nn.Parameter(feats)
        self.voxels[scene_name]['img_feat_grid'] = img_feat_grid
        self.voxels[scene_name]['updated_depth'] = torch.ones(1,1,X,Y,Z).cuda()*999
        self.voxels[scene_name]['img_feat_grid'] = self.voxels[scene_name]['img_feat_grid'].cpu()

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def density_total_variation_add_grad(self, scene_name, weight, dense_mode):
        w = weight * self.voxels[scene_name]['density'].world_size.max() / 128
        self.voxels[scene_name]['density'].total_variation_add_grad(w, 0, w, dense_mode)

    def k0_total_variation_add_grad(self, scene_name, weight, dense_mode):
        w = weight * self.voxels[scene_name]['feature'].world_size.max() / 128
        self.voxels[scene_name]['feature'].total_variation_add_grad(w, w, w, dense_mode)

    @torch.no_grad()
    def maskout_near_cam_vox(self, density, xyz_min, xyz_max, world_size, cam_o, near_clip):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(xyz_min[0], xyz_max[0], world_size[0]),
            torch.linspace(xyz_min[1], xyz_max[1], world_size[1]),
            torch.linspace(xyz_min[2], xyz_max[2], world_size[2]),
        indexing='ij'), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        density.grid[nearest_dist[None,None] <= near_clip] = -100
        return density

    @torch.no_grad()
    def update_occupancy_cache(self, scene_name, xyz_min, xyz_max):
        mask_cache = self.voxels[scene_name]['mask_cache']
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(xyz_min[0], xyz_max[0], mask_cache.mask.shape[0]),
            torch.linspace(xyz_min[1], xyz_max[1], mask_cache.mask.shape[1]),
            torch.linspace(xyz_min[2], xyz_max[2], mask_cache.mask.shape[2]),
        indexing='ij'), -1).cuda()
        cache_grid_density = self.voxels[scene_name]['density'](cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.voxels[scene_name]['mask_cache'].mask &= (cache_grid_alpha > self.fast_color_thres)

    def sample_ray(self, rays_o, rays_d, near, far, xyz_min, xyz_max, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            xyz_min, xyz_max  the xyz_min and xyz_max distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size

        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, xyz_min, xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id, N_steps, t_min, t_max, stepdist, mask_inbbox

    def export_grid(self, scene_name, save_path=None, scale=1):
        voxel = self.voxels[scene_name]
        density = voxel['density'].get_dense_grid().cuda()
        feature = voxel['feature'].get_dense_grid().cuda()[:,:3]
        if scale > 1:
            feature = F.interpolate(feature, scale_factor=scale, mode='trilinear', align_corners=False)
            density = F.interpolate(density, scale_factor=scale, mode='trilinear', align_corners=False)
        alpha = self.activate_density(density).squeeze().cpu().detach().numpy()
        rgb = torch.sigmoid(feature).squeeze().permute(1,2,3,0).cpu().detach().numpy()
        if save_path is not None:
            np.savez_compressed(save_path, alpha=alpha, rgb=rgb)
        ret_dict = {
            'alpha': alpha,
            'rgb': rgb
        }
        return ret_dict

    def save_grid(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.voxels, f)

    def load_grids(self, path, data_list):
        multiprocessing.set_start_method('spawn', force=True)
        scene_list = set([data['scene_name'] for data in data_list])
        ckpt_paths = [os.path.join(path, f) for f in os.listdir(path) if f.split('.')[0] in scene_list]

        print("Load Pretrained Grid")
        jobs = []
        manager = multiprocessing.Manager()
        ret_voxels_dict = manager.dict()
        path_len = len(ckpt_paths)
        split = np.array_split(range(path_len), 16)
        for i in range(16):
            p = multiprocessing.Process(target=self._load_grid, args=(split[i], ckpt_paths, ret_voxels_dict))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()

        self.voxels = dict(ret_voxels_dict)
        print(f"Done: {len(self.voxels.keys())} scenes")
        multiprocessing.set_start_method('fork', force=True)

    def _load_grid(self, split, paths, ret_dict):
        for i in split:
            ckpt_p = paths[i]
            with open(ckpt_p, 'rb') as f:
                scene_name = ckpt_p[-14:-4]
                voxel = pickle.load(f)
                voxel[scene_name]['density'] = voxel[scene_name]['density'].cpu()
                voxel[scene_name]['feature'] = voxel[scene_name]['feature'].cpu()
                voxel[scene_name]['mask_cache'] = voxel[scene_name]['mask_cache'].cpu()
                ret_dict.update(voxel)

    def load_grid(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.voxels = pickle.load(f)

    def scale_grid(self, world_size, scene_name):
        self.voxels[scene_name]['density'].scale_volume_grid(world_size)
        self.voxels[scene_name]['feature'].scale_volume_grid(world_size)
        self.voxel_size *= 0.5
        self.voxel_size_ratio = self.voxel_size / self.final_voxel_size

    @torch.no_grad()
    def query_ray(self, scene_name, rays_o, rays_d, **render_kwargs):
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'
        assert scene_name in self.voxels.keys(), f'Voxel of {scene_name} is not initalized'
        voxel = self.voxels[scene_name]
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id, N_steps, t_min, t_max, stepdist, mask_inbbox = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)

        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        if render_kwargs['topdown']:
            height_mask = ray_pts[:,1]>=0
            ray_pts = ray_pts[height_mask]
            ray_id = ray_id[height_mask]
            step_id = step_id[height_mask]

        if voxel['mask_cache'] is not None:
            mask = voxel['mask_cache'].cuda()(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        
        # query for alpha w/ post-activation
        density = voxel['density'].cuda()(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        n_step = segment_coo(
                src=(weights * step_id),
                index=ray_id,
                out=torch.zeros([N], device=rays_o.device),
                reduce='sum')
        
        alpha_a = segment_coo(
                src=(alpha**2),
                index=ray_id,
                out=torch.zeros([N], device=rays_o.device),
                reduce='mean')
        alpha_b = segment_coo(
                src=(alpha),
                index=ray_id,
                out=torch.zeros([N], device=rays_o.device),
                reduce='mean')**2
        alpha_var = alpha_a-alpha_b
        depth = t_min + stepdist * n_step
        # python pretrain.py --pretrain_scene scene-0016 --bsz 1 --nworkers 0 --config ./configs/pretrain.py
        ret_dict = {
            'depth': depth,
            'var': alpha_var,
        }
        
        return ret_dict

    def forward(self, scene_name, rays_o, rays_d, viewdirs, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'
        assert scene_name in self.voxels.keys(), f'Voxel of {scene_name} is not initalized'
        img_feats = None
        voxel = self.voxels[scene_name]

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id, N_steps, t_min, t_max, stepdist, mask_inbbox = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        
        if voxel['mask_cache'] is not None:
            mask = voxel['mask_cache'].cuda()(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        if render_kwargs['topdown']:
            height_mask = ray_pts[:,1]>=0
            ray_pts = ray_pts[height_mask]
            ray_id = ray_id[height_mask]
            step_id = step_id[height_mask]

        # query for alpha w/ post-activation
        density = voxel['density'].cuda()(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        k0 = voxel['feature'].cuda()(ray_pts)
        if 'img_feat_grid' in voxel.keys():
            img_feats = voxel['img_feat_grid'].cuda()(ray_pts)
        if self.rgbnet is None:
            # no view-depend effect
            rgb_logit = eval_sh(self.sh_base, k0.view(k0.shape[0], 3, (self.sh_base+1)**2), viewdirs[ray_id])
            rgb = torch.sigmoid(rgb_logit)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]

            k0_view = k0[:, 3:]
            k0_diffuse = k0[:, :3]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
            if img_feats is not None:
                rgb_feat = torch.cat([rgb_feat, img_feats], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit + k0_diffuse)
        
        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3], device=rgb.device),
                reduce='sum')

        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
        })

        if render_kwargs.get('render_depth', False):
            n_step = segment_coo(
                src=(weights * step_id),
                index=ray_id,
                out=torch.zeros([N], device=rgb.device),
                reduce='sum')
            depth = t_min + stepdist * n_step
            ret_dict.update({'depth_marched': depth})

        return ret_dict


''' Misc
'''
class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None

class Raw2Alpha_nonuni(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        exp, alpha = render_utils_cuda.raw2alpha_nonuni(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_nonuni_backward(exp, grad_back.contiguous(), interval), None, None

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None

