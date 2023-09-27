from distutils.command.config import config
from dataloader.dataloader import MVMapDataset, MVMapDataset_NeRF, MVMapDataset_Fusion
import torch

def build_dataloader(cfg, args, is_train=True, is_online=True, load_eval=True, shuffule_train=True):
    train_loader = None
    if is_train:
        train_dataset = MVMapDataset(cfg=cfg,
                                        nusc_version=args.nusc_version,
                                        nusc_dataroot=args.nusc_dataroot,
                                        nerf_dataroot=args.nerf_dataroot,
                                        hdmap_dataroot=args.hdmap_dataroot,
                                        is_mini_version=args.debug,
                                        is_online=is_online,
                                        is_presave=args.presave,
                                        is_train=True)
        if args.multicard:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=shuffule_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                        batch_size=args.bsz, 
                                                        sampler=train_sampler,    
                                                        num_workers=args.nworkers,
                                                        pin_memory=True,  
                                                        drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=args.bsz, 
                                                shuffle=shuffule_train,  
                                                pin_memory=True, 
                                                num_workers=args.nworkers, 
                                                drop_last=True)
        if not load_eval:
            return train_loader

    val_dataset = MVMapDataset(cfg=cfg,
                                    nusc_version=args.nusc_version,
                                    nusc_dataroot=args.nusc_dataroot,
                                    nerf_dataroot=args.nerf_dataroot,
                                    hdmap_dataroot=args.hdmap_dataroot,
                                    is_mini_version=args.debug,
                                    is_online=is_online,
                                    is_train=False)
    
    
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=args.bsz, 
                                             shuffle=False, 
                                             pin_memory=True,
                                             num_workers=args.nworkers,
                                             drop_last=False)
    return train_loader, val_loader
    
def build_pretrain_loader(cfg, args):
    nerf_dataset = MVMapDataset_NeRF(cfg=cfg,
                                        nusc_version=args.nusc_version,
                                        nusc_dataroot=args.nusc_dataroot,
                                        nerf_dataroot=args.nerf_dataroot,
                                        hdmap_dataroot=args.hdmap_dataroot,
                                        pretrain_scene=args.pretrain_scene)
    nerf_loader = torch.utils.data.DataLoader(nerf_dataset, 
                                               batch_size=args.bsz, 
                                               shuffle=True, 
                                               pin_memory=True, 
                                               num_workers=args.nworkers)
    return nerf_loader


def build_fusion_loader(cfg, args, is_train=True):
    if is_train:
        fusion_train_dataset = MVMapDataset_Fusion(cfg=cfg,
                                                        nusc_version=args.nusc_version,
                                                        nusc_dataroot=args.nusc_dataroot,
                                                        nerf_dataroot=args.nerf_dataroot,
                                                        hdmap_dataroot=args.hdmap_dataroot,
                                                        is_mini_version=args.debug,
                                                        is_train=True)
        fusion_train_loader = torch.utils.data.DataLoader(fusion_train_dataset, 
                                                        batch_size=args.bsz, 
                                                        shuffle=False, 
                                                        pin_memory=True, 
                                                        num_workers=args.nworkers)
        return fusion_train_loader
    else:
        fusion_val_dataset = MVMapDataset_Fusion(cfg=cfg,
                                                    nusc_version=args.nusc_version,
                                                    nusc_dataroot=args.nusc_dataroot,
                                                    nerf_dataroot=args.nerf_dataroot,
                                                    hdmap_dataroot=args.hdmap_dataroot,
                                                    is_mini_version=args.debug,
                                                    is_train=False)
        fusion_val_loader = torch.utils.data.DataLoader(fusion_val_dataset, 
                                                        batch_size=args.bsz, 
                                                        shuffle=False, 
                                                        pin_memory=True, 
                                                        num_workers=args.nworkers)
        return fusion_val_loader

