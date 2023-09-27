import numpy as np
import torch
from utils.misc import time_once
import time

''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device), indexing='ij')  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.

    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc=False, inverse_y=False, flip_x=False, flip_y=False, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, depth_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    return rgb_tr, depth_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, depth_tr_ori, mask_tr_ori, train_poses, HW, Ks):
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    depth_tr = None
    mask_tr = None
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    train_poses = train_poses.to(DEVICE)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    if depth_tr_ori is not None:
        depth_tr = torch.zeros([N], device=DEVICE)
    if mask_tr_ori is not None:
        mask_tr = torch.zeros([N], device=DEVICE).bool()
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    if depth_tr_ori is None:
        depth_tr_ori = [None] * len(rgb_tr_ori)

    for c2w, img, dep, mask, (H, W), K in zip(train_poses, rgb_tr_ori, depth_tr_ori, mask_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        H, W = int(H.item()), int(W.item())
        rays_o, rays_d, viewdirs = get_rays_of_a_view(H=H, W=W, K=K, c2w=c2w)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        if dep is not None:
            depth_tr[top:top+n].copy_(dep.flatten(0))
        if mask is not None:
            mask_tr[top:top+n].copy_(mask.flatten(0))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    return rgb_tr, depth_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

def gather_training_rays(cfg, data_dict, device):
    depth_tr_ori = None
    rgb_tr_ori = data_dict['images'][0]
    mask_tr_ori = data_dict['masks'][0]
    if cfg.depth_supervise:
        depth_tr_ori = data_dict['depths'][0]

    rgb_tr, depth_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays_flatten(
        rgb_tr_ori=rgb_tr_ori,
        depth_tr_ori=depth_tr_ori,
        mask_tr_ori=mask_tr_ori,
        train_poses=data_dict['poses'][0],
        HW=data_dict['HW'][0], 
        Ks=data_dict['Ks'][0])

    index_generator = batch_indices_generator(rgb_tr.shape[0], cfg.N_rays_per_iter)
    batch_index_sampler = lambda: next(index_generator)
    return rgb_tr, depth_tr, mask_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler


def batch_indices_generator(N, BS):
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

"""
 Projects world points to image pixels.
"""
@torch.no_grad()
def project_world2img(
        world_points,       # [BS*N*4] in the world coordination 
        cam2world,          # [BS*4*4] cam to world projection metrix
        cam2pix,            # [BS*3*3] cam to pix intrinsics 
        size,               # [2]   image shape [HxW]
        ret_masked = False, # bool  return masked pix or return mask itself
        ret_depth = False,  # bool  return depth value
        threshold = 9999,   # int depth threshold
    ):                      # Return: pixel coords [N, h, w]
    H, W = size
    BS, _, _ = cam2world.shape
    w2cs = torch.inverse(cam2world)
    if len(world_points.shape) == 2:
        world_points = world_points.unsqueeze(0).repeat_interleave(BS, 0)
    else:
        B, N, _ = world_points.shape
        try:
            world_points = world_points.unsqueeze(1).repeat_interleave(BS//B, 0).view(BS,N,4)
        except RuntimeError:
            world_points = world_points.unsqueeze(1).repeat_interleave(BS//B, 0).view(BS,N,4)
    cam_coords = w2cs @ world_points.permute(0,2,1)
    depth = cam_coords[:,2,:].unsqueeze(1)
    back_mask = (depth>0)
    EPS = 1e-6
    cam_coords = cam_coords/torch.clamp(depth, min=EPS)
    cam_coords = cam_coords[:, :3, :]
    pix_coords = (cam2pix @ cam_coords)[:, :2,:]

    x_mask = (pix_coords[:,0] > 0) & (pix_coords[:,0] < W)
    y_mask = (pix_coords[:,1] > 0) & (pix_coords[:,1] < H)
    bound_mask = (x_mask & y_mask)
    mask = back_mask[:,0] & bound_mask
    mask &= depth[:,0]<=threshold
    
    if ret_masked:
        img_pix = pix_coords.permute(0,2,1)[mask].flip(-1)
        valid_depth = depth[:,0][mask]

        if ret_depth:
            return img_pix, valid_depth
        return img_pix

    img_pix = pix_coords.permute(0,2,1).flip(-1)
    if ret_depth:
        return img_pix, mask, depth[:,0]
    return img_pix, mask
    

    
def project_img2world(
    pix_coords,         # [N*3] in the world coordination 
    cam2world,          # [4*4] cam to world projection metrix
    cam2pix,            # [3*3] cam to pix intrinsics 
    depth = 5           # int,  project distance
):                      # Return: pixel coords [N, h, w]
    pix2cam = torch.inverse(cam2pix)
    cam_coords = (pix2cam @ pix_coords.T).T

    cam_coords *= depth
    cam_coords = torch.cat([cam_coords, torch.ones((cam_coords.shape[0],1)).cuda()], dim=1)
    world_coords = (cam2world @ cam_coords.T).T[:,:3]

    return world_coords