import random, os
import numpy as np
import torch
import torch_scatter
import time
import subprocess
import torch.distributed as dist

mse2psnr = lambda x : -10. * torch.log10(x)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

def masked_softmax(vec, mask, dim=0, is_masked=True):
    if not is_masked:
        masked_vec = vec * mask.float()
    else:
        masked_vec = vec
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec-max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros=(masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/masked_sums

def load_ckpt(model, args, cfg, logger, gpu=0, opt=None, sched=None):
    base_dir = os.path.join(args.logdir, cfg.expname)
    ckpts = [ckpt for ckpt in os.listdir(base_dir) if ckpt.split('.')[-1] == 'pth']
    latest_ckpt = sorted(ckpts, key=lambda x: float(x.split('_')[-1][:-4]))[-1] # First Epoch
    ckpt_path = os.path.join(base_dir, latest_ckpt)
    if args.multicard:
        model_dict = torch.load(ckpt_path, map_location='cuda:{}'.format(gpu))
        model.module.load_state_dict(model_dict['model'])
    else: 
        model_dict = torch.load(ckpt_path, map_location='cuda:0')
        for key in list(model_dict['model'].keys()):
            if 'fuser' in key:
                model_dict['model'].pop(key)
        model.load_state_dict(model_dict['model'], strict=False)
        logger.info(f"Checkpoint Loaded: {ckpt_path}")
        
    if opt is not None:
        opt.load_state_dict(model_dict['optimizer'])
    if sched is not None:
        sched.load_state_dict(model_dict['scheduler'])
    start = 0 if not 'last_epoch' in model_dict.keys() else int(model_dict['last_epoch']) + 1
    return model, opt, sched, start

def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = torch.zeros((num_classes, H, W))
    onehot.scatter_(0, label[None].long(), 1)
    return onehot

def batch_indices_generator(N, BS):
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        end_flag = False
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
            end_flag = True
        yield (idx[top:top+BS], end_flag)
        top += BS

def export_grid(path, model, scene_name):
    with torch.no_grad():
        alpha = model.nerf_model.activate_density(model.nerf_model.voxels[scene_name]['density'].get_dense_grid()).squeeze().cpu().detach().numpy()
        X, Y, Z = alpha.shape
        alpha = alpha[:, :(Y//2), :]
        rgb = torch.sigmoid(model.nerf_model.voxels[scene_name]['feature'].get_dense_grid()[:,:3,:,:(Y//2),:]).squeeze().permute(1,2,3,0).cpu().detach().numpy()
    np.savez_compressed(path, alpha=alpha, rgb=rgb)

def get_render_kwargs(cfg, data_dict, topdown=False, render_depth=True, idx=None):
    render_kwargs = {
        'near': data_dict['near'].float().item() if idx is None else data_dict['near'][idx].float().item(),
        'far': data_dict['far'].float().item() if idx is None else data_dict['far'][idx].float().item(),
        'xyz_min': data_dict['xyz_min'][0].cuda() if idx is None else data_dict['xyz_min'][idx].cuda(),
        'xyz_max': data_dict['xyz_max'][0].cuda() if idx is None else data_dict['xyz_max'][idx].cuda(),
        'stepsize': cfg.main_cfg.stepsize,
        'render_depth': render_depth,
        'topdown': topdown
    }
    return render_kwargs

def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

def weights_init(m):
    import torch.nn as nn
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_corr(w, feats):
    B, N = feats.shape[:2]
    feats = feats.permute(1,0,2,3)
    x, feats = w.view(B, -1), feats.view(N,B,-1)

    corrs = []
    for y in feats:
        dim = -1
        centered_x = x - x.mean(dim=dim, keepdim=True)
        centered_y = y - y.mean(dim=dim, keepdim=True)

        covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

        bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

        x_std = x.std(dim=dim, keepdim=True)
        y_std = y.std(dim=dim, keepdim=True)

        corr = bessel_corrected_covariance / (x_std * y_std)
        corrs.append(corr)
    corrs = torch.stack(corrs).squeeze(-1)
    return corrs

def get_batch_iou(pred_map, gt_map):
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool()
        gt_map = gt_map.bool()

        for i in range(pred_map.shape[1]):
            pred = pred_map[:, i]
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum().float()
            union = (pred | tgt).sum().float()
            intersects.append(intersect)
            unions.append(union)
    return torch.tensor(intersects), torch.tensor(unions)

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


def time_once():
    torch.cuda.synchronize()
    return time.time()


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    assert (deg + 1) ** 2 == sh.shape[-1]
    C = sh.shape[-2]

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                        C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                        C3[1] * xy * z * sh[..., 10] +
                        C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                        C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                        C3[5] * z * (xx - yy) * sh[..., 14] +
                        C3[6] * x * (xx - 3 * yy) * sh[..., 15])
                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def pad_or_trim_to_np(x, shape, pad_val=0):
    shape = np.asarray(shape)
    pad = shape - np.minimum(np.shape(x), shape)
    zeros = np.zeros_like(pad)
    x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
    return x[:shape[0], :shape[1]]

def raval_index(coords, dims):
    dims = torch.cat((dims, torch.ones(1, device=dims.device)), dim=0)[1:]
    dims = torch.flip(dims, dims=[0])
    dims = torch.cumprod(dims, dim=0) / dims[0]
    multiplier = torch.flip(dims, dims=[0])
    indices = torch.sum(coords * multiplier, dim=1)
    return indices

def points_to_voxels(
  points_xyz,
  points_mask,
  grid_range_x,
  grid_range_y,
  grid_range_z
):
    batch_size, num_points, _ = points_xyz.shape
    voxel_size_x = grid_range_x[2]
    voxel_size_y = grid_range_y[2]
    voxel_size_z = grid_range_z[2]
    grid_size = np.asarray([
        (grid_range_x[1]-grid_range_x[0]) / voxel_size_x,
        (grid_range_y[1]-grid_range_y[0]) / voxel_size_y,
        (grid_range_z[1]-grid_range_z[0]) / voxel_size_z
    ]).astype('int32')
    voxel_size = np.asarray([voxel_size_x, voxel_size_y, voxel_size_z])
    voxel_size = torch.Tensor(voxel_size).to(points_xyz.device)
    num_voxels = grid_size[0] * grid_size[1] * grid_size[2]
    grid_offset = torch.Tensor([grid_range_x[0], grid_range_y[0], grid_range_z[0]]).to(points_xyz.device)
    shifted_points_xyz = points_xyz - grid_offset
    voxel_xyz = shifted_points_xyz / voxel_size
    voxel_coords = voxel_xyz.int()
    grid_size = torch.from_numpy(grid_size).to(points_xyz.device)
    grid_size = grid_size.int()
    zeros = torch.zeros_like(grid_size)
    voxel_paddings = ((points_mask < 1.0) |
                      torch.any((voxel_coords >= grid_size) |
                                (voxel_coords < zeros), dim=-1))
    voxel_indices = raval_index(
      torch.reshape(voxel_coords, [batch_size * num_points, 3]), grid_size)
    voxel_indices = torch.reshape(voxel_indices, [batch_size, num_points])
    voxel_indices = torch.where(voxel_paddings,
                                torch.zeros_like(voxel_indices),
                                voxel_indices)
    voxel_centers = ((0.5 + voxel_coords.float()) * voxel_size + grid_offset)
    voxel_coords = torch.where(torch.unsqueeze(voxel_paddings, dim=-1),
                               torch.zeros_like(voxel_coords),
                               voxel_coords)
    voxel_xyz = torch.where(torch.unsqueeze(voxel_paddings, dim=-1),
                            torch.zeros_like(voxel_xyz),
                            voxel_xyz)
    voxel_paddings = voxel_paddings.float()

    voxel_indices = voxel_indices.long()
    points_per_voxel = torch_scatter.scatter_sum(
        torch.ones((batch_size, num_points), dtype=voxel_coords.dtype, device=voxel_coords.device) * (1-voxel_paddings),
        voxel_indices,
        dim=1,
        dim_size=num_voxels
    )

    voxel_point_count = torch.gather(points_per_voxel,
                                     dim=1,
                                     index=voxel_indices)


    voxel_centroids = torch_scatter.scatter_mean(
        points_xyz,
        voxel_indices,
        dim=1,
        dim_size=num_voxels)
    point_centroids = torch.gather(voxel_centroids, dim=1, index=torch.unsqueeze(voxel_indices, dim=-1).repeat(1, 1, 3))
    local_points_xyz = points_xyz - point_centroids

    result = {
        'local_points_xyz': local_points_xyz,
        'shifted_points_xyz': shifted_points_xyz,
        'point_centroids': point_centroids,
        'points_xyz': points_xyz,
        'grid_offset': grid_offset,
        'voxel_coords': voxel_coords,
        'voxel_centers': voxel_centers,
        'voxel_indices': voxel_indices,
        'voxel_paddings': voxel_paddings,
        'points_mask': 1 - voxel_paddings,
        'num_voxels': num_voxels,
        'grid_size': grid_size,
        'voxel_xyz': voxel_xyz,
        'voxel_size': voxel_size,
        'voxel_point_count': voxel_point_count,
        'points_per_voxel': points_per_voxel
    }


    return result
