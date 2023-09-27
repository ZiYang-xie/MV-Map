import numpy as np
import os, imageio
from functools import reduce

from pyquaternion import Quaternion

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

def load_nuscenes(basedir,  
                  load_imgs=False, 
                  load_depths=False,
                  near=0,
                  far=100,
                  N_cam=6,
                  width=1600,
                  height=900):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    if poses_arr.shape[1] == 19:
        poses = poses_arr[:,:-4].reshape([-1, 3, 5]).transpose([1,2,0])
        poses_arr = poses_arr[:,:-2]
    if poses_arr.shape[1] == 17:
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    elif poses_arr.shape[1] == 14:
        poses = poses_arr[:, :-2].reshape([-1, 3, 4]).transpose([1,2,0])
    else:
        raise NotImplementedError
    bds = poses_arr[:, -2:].transpose([1,0])

    factor = 1
    # Load Intrinsics
    raw_cam_K = poses[:,4,:].copy().astype(np.float32).transpose([1,0])
    raw_cam_K = raw_cam_K/factor
    cx = raw_cam_K[:,0]
    cy = raw_cam_K[:,1]
    focal = raw_cam_K[:,2]
    Ks = [np.array([[focal[i],0,cx[i]],[0,focal[i],cy[i]],[0,0,1]]) for i in range(len(raw_cam_K))]
    Ks = np.stack(Ks, 0)

    imgfiles = np.load(os.path.join(basedir, 'im_paths.npy'))
    assert poses.shape[-1] == len(imgfiles)

    sh = [height, width]
    if poses.shape[1] == 4:
        poses = np.concatenate([poses, np.zeros_like(poses[:,[0]])], 1)
        poses[2, 4, :] = np.load(os.path.join(basedir, 'hwf_cxcy.npy'))[2]
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor

    imgs = None
    if load_imgs:
        imgs = [imageio.imread(f)[...,:3]/255. for f in imgfiles]
        imgs = np.stack(imgs)
    else:
        imgs = imgfiles

    depths = None
    depths_file = None
    render_depth_files = None
    if os.path.exists(os.path.join(basedir, 'depths')):
        depths_file = [os.path.join(basedir, 'depths', f) for f in sorted(os.listdir(os.path.join(basedir, 'depths')))]
    if load_depths:
        depths = [imageio.imread(f) / 256 for f in depths_file]

        for idx, depth in enumerate(depths):
            if(depth.shape[0] == sh[0]):
                continue
            depths[idx] = np.concatenate([depths[-1],np.zeros((sh[0]-depth.shape[0],sh[1]))])

        depths = np.stack(depths).astype(np.float32)[0]
    elif depths_file is not None:
        depths = depths_file

    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # [-u, r, -t] -> [r, u, -t]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)


    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    HW = np.array([[height, width]]*len(imgs))

    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]

    render_poses = []

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far, near_clip=0,
        poses=poses, render_poses=render_poses,
        images=imgs, depths=depths
    )
    return data_dict


def get_lidar_data(nusc, ref_sd_token, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    #ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = ref_sd_token
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points