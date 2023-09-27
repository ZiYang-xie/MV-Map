from model.voxel_encoder import VoxelEncoder
from model.bev_decoder import BevDecoder
from model.image_encoder import ImgEncoder
from model.temporal_fuser import Fuser
import torch

'''
Build NeRF Model for HDMap generation
'''
def build_nerf_model(cfg, data_list):
    from model.voxel_nerf import VoxelNeRF
    if cfg.model_type == 'direct_voxel':
        model = VoxelNeRF(cfg, data_list)
        if cfg.ckpt_path != None:
            model.load_grids(cfg.ckpt_path, data_list)
        return model
    else:
        raise NotImplementedError

def build_encoder(cfg):
    return VoxelEncoder(cfg)

def build_decoder(cfg):
    return BevDecoder(cfg)

def build_fuser(cfg):
    return Fuser(cfg)

def build_img_encoder(cfg, device):
    return ImgEncoder(cfg, device)

def build_pc_encoder(cfg):
    from model.pointpillar import PointPillarEncoder
    return PointPillarEncoder(cfg)

def build_nerf_optimizer(cfg, model, pretrained=False):
    param_group = []
    if model.nerf_model.rgbnet is not None:
        param_group.append({'params': model.nerf_model.rgbnet.parameters(), 'lr': cfg.nerf_opt.lr, 'skip_zero_grad': False})
    # param_group.append({'params': model.image_encoder.parameters()})

    for key, voxel in model.nerf_model.voxels.items():
        param_group.append({'params': voxel['density'].parameters(), 
                            'lr': cfg.nerf_opt.density_lr if pretrained else cfg.nerf_opt.density_lr_finetune, 
                            'skip_zero_grad': True})
        param_group.append({'params': voxel['feature'].parameters(), 
                            'lr': cfg.nerf_opt.feature_lr if pretrained else cfg.nerf_opt.feature_lr_finetune, 
                            'skip_zero_grad': True})
    
    nerf_opt = torch.optim.AdamW(params=param_group)
    return nerf_opt

def build_main_optimizer(cfg, model, max_iters):
    # Build Main optimizer
    opt_type = cfg.main_opt.pop('type')
    if opt_type == 'SGD':
        opt = torch.optim.SGD  
    elif opt_type == 'Adam':
        opt = torch.optim.Adam
    elif opt_type == 'AdamW':
        opt = torch.optim.AdamW
    else:
        raise NotImplementedError
    param_group = []
    param_group.append({'params': model.image_encoder.parameters()})
    if getattr(model, 'pc_encoder', None) is not None:
        param_group.append({'params': model.pc_encoder.parameters()})
    param_group.append({'params': model.voxel_encoder.parameters()})
    param_group.append({'params': model.bev_decoder.parameters()})
    main_opt = opt(params=param_group, **cfg.main_opt)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(main_opt, cfg.main_opt.lr, max_iters+100, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return main_opt, scheduler

def build_fusion_optimizer(cfg, model, max_iters):
    opt_type = cfg.fuser_opt.pop('type')
    fuse_type = cfg.fuser_opt.pop('fuse_type')
    if opt_type == 'SGD':
        opt = torch.optim.SGD  
    elif opt_type == 'Adam':
        opt = torch.optim.Adam
    elif opt_type == 'AdamW':
        opt = torch.optim.AdamW
    else:
        raise NotImplementedError
    param_group = []
    param_group.append({'params': model.fuser.parameters()})
    if fuse_type == 'feature':
        param_group.append({'params': model.bev_decoder.parameters(), 'lr': 2e-5})
    fusion_opt = opt(params=param_group, **cfg.fuser_opt)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(fusion_opt, cfg.fuser_opt.lr, max_iters+100, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return fusion_opt, scheduler

def build_optimizer(cfg, model):
    nerf_opt = build_nerf_optimizer(cfg, model)
    main_opt = build_main_optimizer(cfg, model)
    return main_opt, nerf_opt
