import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.factory import build_nerf_model, \
                          build_fuser, \
                          build_encoder, \
                          build_decoder, \
                          build_img_encoder, \
                          build_pc_encoder

from dataloader.constant import CAMS
from scipy.spatial.transform import Rotation as R
from utils.map_utils import get_crop_center
from utils.misc import get_render_kwargs, time_once, onehot_encoding, get_corr, masked_softmax
from utils.ray_utils import project_world2img, project_img2world, get_rays_of_a_view
from utils.voxel_utils import flatten_sample_coords, reduce_masked_mean

class MVMap(nn.Module):
    def __init__(self,
                 cfg,
                 data_list: list,
                 device):
        super(MVMap, self).__init__()
        self.cfg = cfg
        self.N_cams = cfg.main_cfg.data_config.N_cams
        self.device = device
        self.voxel_size = self.cfg.main_cfg.data_config.gt_voxel_size
        self.final_voxel_size = self.cfg.final_voxel_size
        self.image_encoder = build_img_encoder(cfg.img_encoder_config, device)
        if cfg.nerf_cfg.ckpt_path is None:
            self.nerf_model = build_nerf_model(cfg.nerf_cfg, data_list)
        self.voxel_encoder = build_encoder(cfg.encoder_config)
        self.bev_decoder = build_decoder(cfg.decoder_config)
        self.crop_base = self.generate_patch_coords(cfg.main_cfg)
        self.sample_coords = None

        # IF LiDAR, build pc encoder
        self.use_lidar = cfg.main_cfg.data_config.use_lidar
        self.use_image = getattr(cfg.main_cfg.data_config, 'use_image', True)

        assert self.cfg.main_cfg.fuse_type in ['feature', 'logits']
        self.fuse_type = self.cfg.main_cfg.fuse_type
        self.fuser = build_fuser(cfg.fuser_config)

    def init_global_map(self, data_list, channels):
        global_feat_map = None
        for data_dict in data_list:
            scene_name = data_dict['scene_name']
            global_size = data_dict['global_size']
            global_feat_map = self.init_single_scene(channels, global_size)
            break
        
        return global_feat_map

    def init_single_scene(self, channels, global_size):
        world_size = torch.tensor([global_size[0], global_size[-1]])
        scene_map = nn.Parameter(torch.zeros([1, channels, *world_size]).to(self.device))
        scene_map.requires_grad = True
        return scene_map
        
    def generate_patch_coords(self, cfg):
        v_s = self.voxel_size
        h_range = cfg.data_config.height_range
        h_res = int((h_range[1] - h_range[0]) * cfg.data_config.hight_resolution_scale)
        patch_sz = cfg.data_config.patch_size
        p_res = int((patch_sz[1] - patch_sz[0])/v_s * cfg.data_config.sample_resolution_scale)
        crop_base = torch.stack(torch.meshgrid(
                                    torch.linspace(int(h_range[0]), int(h_range[1]), h_res, device=self.device),
                                    torch.linspace(int(patch_sz[0])+1, int(patch_sz[1]), p_res, device=self.device), 
                                    torch.linspace(int(patch_sz[0])+1, int(patch_sz[1]), p_res, device=self.device) 
                                    #torch.linspace(int(-15)+1, int(15), 200, device=self.device) 
                                ,indexing='ij')).permute(1,2,3,0).contiguous()

        crop_base = torch.stack([
                        crop_base[...,1],
                        crop_base[...,2],
                        crop_base[...,0]
                    ],dim=3)
        return crop_base

    @torch.no_grad()
    def query_ray_probability(self, index, scene_name, Ps, Ks, ret_dict, render_kwargs):
        mask = ret_dict['mask'][index]
        pixels = ret_dict['pixels'][index][mask]
        rays_o = Ps[index, :,:3,3].unsqueeze(1).repeat_interleave(mask.shape[1],1)
        c2ws = Ps[index, :,:3,:3].unsqueeze(1).repeat_interleave(mask.shape[1],1)
        intrs = Ks[index].unsqueeze(1).repeat_interleave(mask.shape[1],1)
        rays_o = rays_o[mask]
        c2ws = c2ws[mask]
        intrs = intrs[mask]
        dirs = torch.stack([(pixels[:,1]-intrs[:,0,2])/intrs[:,0,0], -(pixels[:,0]-intrs[:,1,2])/intrs[:,1,1], -torch.ones(pixels.shape[0], device=pixels.device)], -1)
        rays_d = torch.sum((dirs.unsqueeze(1) * c2ws), -1)

        stop_weights = self.nerf_model.query_ray(scene_name[index], rays_o, rays_d, **render_kwargs)
        return stop_weights


    def _feats_sampling(self, img_feats, data_dict, render_depth, ori_coords, sample_coords, Ps, Ks, H, W, far):
        Y, X, Z, _ = ori_coords.shape
        BS, C, X_v, Z_v = img_feats.shape
        B = BS//self.N_cams

        scale_factor = torch.tensor([H,W], device=img_feats.device)
        #* Project Points
        pix_c, mask, depth = project_world2img(world_points=sample_coords, 
                                                cam2world=Ps, 
                                                cam2pix=Ks, 
                                                size=(H,W),
                                                ret_depth=True)
        pix_c = torch.nan_to_num(pix_c, -1)
        N = pix_c.shape[1]

        samp_ind = (pix_c / scale_factor) * 2 - 1
        samp_ind = samp_ind.view(BS, Y, X, Z, 2)
        z_pix = torch.zeros((*samp_ind.shape[:-1], 1), device=samp_ind.device)
        samp_ind = torch.cat([samp_ind,z_pix], dim=4)

        #* Sample Feature from Image feats
        out_feats = F.grid_sample(img_feats.permute(0,1,3,2).unsqueeze(2), samp_ind, mode='bilinear', align_corners=False)
        
        BS,C,Y,X,Z = out_feats.shape
        mask = mask.view(B,self.N_cams,1,Y,X,Z).int()
        out_feats = out_feats.view(B, self.N_cams, C, Y, X, Z)
        out_feats = reduce_masked_mean(out_feats, mask, dim=1, keepdim=False)
        out_feats = out_feats.view(B, C*Y, X, Z)

        #* NeRF Info
        ret_info = {}
        if getattr(self.cfg.fuser_config, 'use_var_info', False):
            variances = data_dict['variances'][...,0].to(self.device).view(BS, 1, H, W).float()
            var_info = F.grid_sample(variances.permute(0,1,3,2).unsqueeze(2), samp_ind, mode='bilinear', align_corners=False)
            var_info = var_info.view(B, self.N_cams, 1, Y, X, Z)
            var_info = reduce_masked_mean(var_info, mask, dim=1, keepdim=False).view(B, Y, X, Z)
            ret_info['var'] = var_info

        if getattr(self.cfg.fuser_config, 'use_depth_info', False):
            render_depth = render_depth.to(self.device).view(BS, 1, H, W).float()
            nerf_depth = F.grid_sample(render_depth.permute(0,1,3,2).unsqueeze(2), samp_ind, mode='bilinear', align_corners=False)
            nerf_depth = nerf_depth.view(B, self.N_cams, 1, Y, X, Z)
            nerf_depth = reduce_masked_mean(nerf_depth, mask, dim=1, keepdim=False)

            depth = depth.view(B, self.N_cams, 1, Y, X, Z)
            depth = reduce_masked_mean(depth, mask, dim=1, keepdim=False)

            ret_info['depth'] = depth
            ret_info['nerf_depth'] = nerf_depth
            
        return out_feats, ret_info

    def lift_feat_to_voxel(self, img_feats, data_dict):
        BS = img_feats.shape[0]
        B = BS//self.N_cams # For temporal fusion, the B = Batch x fusion_frames
        Ps = data_dict['cam2ego'].to(self.device).float().view(BS,4,4)
        Ks = data_dict['Ks'].to(self.device).float().view(BS,3,3)

        render_depth = data_dict['render_depth'] if 'render_depth' in data_dict.keys() else None
        if self.sample_coords is None:
            self.sample_coords = flatten_sample_coords(self.crop_base, B, self.device)

        H, W, _ = data_dict['images'].shape[-3:]
        out_feats, ret_info = self._feats_sampling(img_feats, data_dict, render_depth, self.crop_base, self.sample_coords, Ps, Ks, H, W, data_dict['far'][0])
        return out_feats, ret_info

    def get_img_feats(self, data_dict):
        imgs = data_dict['images'].view(-1, *data_dict['images'].shape[-3:]).permute(0,3,1,2).contiguous()
        img_feats = self.image_encoder(imgs.to(self.device).float())
        return img_feats 
        
    def neural_lifting(self, data_dict):
        voxel_feats, ret_info = None, None
        if self.use_image:
            img_feats = self.get_img_feats(data_dict)
            voxel_feats, ret_info = self.lift_feat_to_voxel(img_feats, data_dict)
        
        return voxel_feats, ret_info

    def get_offsets(self, data_dict, ref_index=None):
        e2g = data_dict['ego2global'][0,:,0].to(self.device)
        B, N = data_dict['ego2global'].shape[:2]
        assert B == 1, 'For now, only support single batch fusion'
        if ref_index is None:
            ref_index = N//2
            
        if self.sample_coords is None:
            self.sample_coords = flatten_sample_coords(self.crop_base, B, self.device)
            
        sample_coords = self.sample_coords.permute(0,2,1).double()
        ref2g = e2g[ref_index]
        g2e = torch.inverse(e2g)
        ref2e = g2e @ ref2g
        #* Warp the target local region to adjacent feature map
        ego_ref_coords = ref2e @ sample_coords
        ego_ref_coords = ego_ref_coords.permute(0,2,1).contiguous()[:,:,:3]
        return ego_ref_coords

    def temporal_warpping(self, feat, data_dict):
        # Wrap the feature
        offset_coords = self.get_offsets(data_dict)
        N = offset_coords.shape[0]
        Y = self.crop_base.shape[0]
        BN, CY, X, Z = feat.shape

        # Get Warp Indices
        max_bound = self.crop_base[-1,-1,-1]
        half_range = (self.crop_base[-1,-1,-1] - self.crop_base[0,0,0]) / 2
        delta = half_range-max_bound
        delta = delta.unsqueeze(0).unsqueeze(0)
        samp_ind = (offset_coords + delta) / half_range
        samp_ind = samp_ind.view(N, Y, X, Z, 3).float()
        
        warpped_feat = F.grid_sample(feat.permute(0,1,3,2), samp_ind[:,0,:,:,:2], mode='bilinear', align_corners=True)
        return warpped_feat
        
    def temporal_fusion(self, data_dict, presaved=False):
        if self.fuse_type == 'feature':
            return self.fuse_feature(data_dict, presaved)
        elif self.fuse_type == 'logits':
            return self.fuse_logits(data_dict, presaved)

    def fuse_feature(self, data_dict, presaved):
        if not presaved:
            with torch.no_grad():
                voxel_feats, ret_info = self.neural_lifting(data_dict)
                bev_feats = self.voxel_encoder(voxel_feats)
                ori_semantic = self.bev_decoder(bev_feats)
        else:
            bev_feats = data_dict['feats'][0].cuda()
            extra_info = {
                'sample_depth': data_dict['depth'][0].squeeze(1),
                'nerf_depth': data_dict['nerf_depth'][0].squeeze(1),
                'var': data_dict['var'][0]
            }
            # depth_info = data_dict['depth_info'][0].cuda() if self.cfg.fuser_config.use_depth_info else None
            ori_semantic = data_dict['ori_semantic'][0].cuda()
        
        C = bev_feats.shape[1]
        weight, pred_kl_div = self.fuser(bev_feats, extra_info)
        warp_batch = torch.cat([bev_feats, weight], dim=1)
        warpped_batch = self.temporal_warpping(warp_batch, data_dict)
        warpped_feats = warpped_batch[:,:C]
        warpped_weight = warpped_batch[:,-1].unsqueeze(1)
        mask = warpped_weight!=0
        warpped_weight = masked_softmax(warpped_weight, mask)
        warpped_feats *= warpped_weight
        fused_feat = warpped_feats.sum(0).unsqueeze(0)
        fused_semantic = self.bev_decoder(fused_feat)

        return fused_semantic, ori_semantic, pred_kl_div

    def fuse_logits(self, data_dict, presaved):
        if not presaved:
            with torch.no_grad():
                voxel_feats, ret_info = self.neural_lifting(data_dict)
                bev_feats = self.voxel_encoder(voxel_feats)
                ori_semantic = self.bev_decoder(bev_feats)
        else:
            bev_feats = data_dict['feats'][0].cuda()
            extra_info = {
                'sample_depth': data_dict['depth'][0].squeeze(1) if self.cfg.fuser_config.use_depth_info else None,
                'nerf_depth': data_dict['nerf_depth'][0].squeeze(1) if self.cfg.fuser_config.use_depth_info else None,
                'var': data_dict['var'][0] if self.cfg.fuser_config.use_var_info else None
            }
            ori_semantic = data_dict['ori_semantic'][0].cuda()

        C = ori_semantic.shape[1]
        weight, pred_kl_div = self.fuser(bev_feats, extra_info)
        warp_batch = torch.cat([ori_semantic, weight], dim=1)
        warpped_batch = self.temporal_warpping(warp_batch, data_dict)
        warpped_feats = warpped_batch[:,:C]
        warpped_weight = warpped_batch[:,-1].unsqueeze(1)
        mask = warpped_weight!=0
        warpped_weight = masked_softmax(warpped_weight, mask)
        warpped_feats *= warpped_weight
        fused_semantic = warpped_feats.sum(0).unsqueeze(0)
        return fused_semantic, ori_semantic, pred_kl_div

    def forward(self, data_dict):
        voxel_feats, ret_info = self.neural_lifting(data_dict)
        bev_feats = self.voxel_encoder(voxel_feats)
        semantic = self.bev_decoder(bev_feats)
        return semantic

def build_model(cfg, data_list, device):
    return MVMap(cfg, data_list, device)