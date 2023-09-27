import cv2
import imageio
from matplotlib import pyplot as plt

import numpy as np
import torch

def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    EPS = 1e-6
    numer = torch.sum(x*mask, dim=dim, keepdim=keepdim)
    denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer/denom
    return mean

def flatten_sample_coords(base_coords, repeat_size, device):
    try:
        sample_coords = base_coords.view(-1,3).unsqueeze(0).repeat_interleave(repeat_size,0)
    except RuntimeError:
        sample_coords = base_coords.view(-1,3).unsqueeze(0).repeat_interleave(repeat_size,0)
    sample_coords = torch.cat([sample_coords, 
                               torch.ones((repeat_size,sample_coords.shape[1],1), 
                               device=device)], dim=-1)
    sample_coords[:,:,3] = 1
    return sample_coords

def export_nerf_mesh(info_dict, save_path, thres=0.01):
    '''
    Input:
        info_dict: {
            'alpha': [Alpha Value]
            'rgb': [RGB Values]
        }
    Output:
        export a ply file to save_path
    '''
    import open3d as o3d
    alpha = info_dict['alpha']
    rgb = info_dict['rgb']
    if rgb.shape[0] < rgb.shape[-1]:
        alpha = np.transpose(alpha, (1,2,0))
        rgb = np.transpose(rgb, (1,2,3,0))

    xyz_min = np.array([0,0,0])
    xyz_max = np.array(alpha.shape)

    xyz = np.stack((alpha > thres).nonzero(), -1)
    crop_alpha = alpha[alpha > thres]
    color = rgb[xyz[:,0], xyz[:,1], xyz[:,2]]
    rgba = np.concatenate((color, crop_alpha[...,None]), axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz / alpha.shape * (xyz_max - xyz_min) + xyz_min)
    pcd.colors = o3d.utility.Vector3dVector(color[:, :3])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
    o3d.io.write_voxel_grid(save_path, voxel_grid)


