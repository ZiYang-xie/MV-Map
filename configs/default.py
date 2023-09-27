expname = 'Default'
ablation_tag = ''
logdir = './logs'

final_voxel_size = 0.5
voxel_size = 0.5 
N_cams = 6

''' Main Configs
'''
main_cfg = dict(
    stepsize=0.5,
    ori_depth=False,
    fusion_frame_num=5,
    fuse_type='logits',
    data_config = dict(
        scene_cata = ['Normal', 'Night', 'Rain'],
        presaved=True,
        presave_path='',
        N_cams = N_cams,
        N_rays_per_iter=16348,
        load_img=False,
        load_depth=False,
        load2gpu_on_the_fly=False,
        depth_supervise=False,
        scale_factor=2.5,
        sample_resolution_scale=1,
        hight_resolution_scale=1,
        near=0.1,
        far=64,
        height_range=[-4,2],
        y_range=[-4,2],
        patch_size=[-50,50],
        gt_voxel_size=0.25,
        voxel_size=voxel_size
    ),
)

''' Loss Configs
'''
loss_cfg = dict (
    gradient_clip=5.0,
    rgb_loss = dict(
        type='mse',
        weight=1,
        weight_rgbper=0.1,
    ),
    tv_loss = dict(
        weight_density=1e-5,
        weight_feature=0,
    ),
    depth_loss = dict(
        type='l1',
        weight=0,
    ),
    feat_loss = dict(weight=0.1),
    aux_loss = dict(weight=0),
    seg_loss = dict(
        type='focal_loss', 
        weights=[0.5,1,1,1],
        pose_weight=2.13,  
        alpha=1,           
        gamma=2,           
    ),
)


''' Optimizer Configs
'''
opt_cfg = dict(
    main_opt = dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=1e-4
    ),
    nerf_opt = dict(
        lr=1e-3,
        lrate_decay=30,
        feature_lr=0.1,
        feature_lr_finetune=0.01,
        density_lr=0.1,
        density_lr_finetune=0.01,
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    fuser_opt = dict(
        type='AdamW',
        fuse_type=main_cfg['fuse_type'],
        lr=1e-3,
    )
)

''' NeRF Configs
'''
nerf_cfg = dict(
    model_type = 'direct_voxel',
    ckpt_path = './logs/pretrained_nerf_trainval',
    voxel_size = voxel_size,
    final_voxel_size = final_voxel_size,
    voxel_config = dict(
        grid_type='DenseGrid',
        feature_dim=12
    ),
    decay_after_scale = 1.0,
    #rgbnet_config = None,
    rgbnet_config = dict (
        view_denpendent=True,
        img_feats_channel=128,
        rgbnet_dim=9,
        rgbnet_direct=False,
        rgbnet_full_implicit=False,
        rgbnet_depth=3,
        rgbnet_width=128,
    ),
    model_config = dict(
        viewbase_pe=4,
        alpha_init=1e-4,
        sh_base=0,
        fast_color_thres=1e-7
    ),
)


''' Image Encoder Configs
'''
img_encoder_config = dict(
    feat_dim=128,
    N_cams=N_cams,
    type=50,
    size=(45,80),
    init_ckpt=None
)

''' Voxel Encoder Configs
'''
encoder_config = dict(
    in_channel=768,
    hidden_channel=256,
    hidden_depth=1,
    out_channel=128
)

''' Temporal fusion Configs
'''
fuser_config = dict(
    fusion_frame_num=main_cfg['fusion_frame_num'],
    use_ori_depth=main_cfg['ori_depth'],
    encoder_type='unet',
    use_depth_info=False,
    uncert_inchannel=128,
    uncert_hidden=128,
    uncert_out_channel=4,
)

''' BEV Decoder Configs
'''
decoder_config = dict(
    type='Simple',
    in_channel=128,     
    out_channel=4,
    instance_seg=False,
    embedded_dim=16
)