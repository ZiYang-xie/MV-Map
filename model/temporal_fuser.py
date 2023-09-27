import torch
import torch.nn as nn
import torch.nn.functional as F
from .bev_decoder import UNet
from utils.misc import weights_init

class Fuser(nn.Module):
    def __init__(self, fuser_cfg):
        super(Fuser, self).__init__()
        uncert_inchannel = fuser_cfg.uncert_inchannel
                
        self.fuser_cfg = fuser_cfg
        self.height_mapper = None
        if getattr(fuser_cfg, 'use_height_mapper', False):
            self.mapper_in_dim = fuser_cfg.mapper_in_dim
            self.mapper_hidden_dim = fuser_cfg.mapper_hidden_dim
            self.mapper_out_dim = fuser_cfg.mapper_out_dim
            self.mapper_layer = fuser_cfg.mapper_layer
            self.height_mapper = nn.Sequential(
                                    nn.Conv2d(self.mapper_in_dim, self.mapper_hidden_dim, 3,1,1),
                                    # nn.BatchNorm2d(self.mapper_hidden_dim),
                                    nn.ReLU(inplace=True),
                                    *[
                                        nn.Conv2d(self.mapper_hidden_dim, self.mapper_hidden_dim,3,1,1),
                                        # nn.BatchNorm2d(self.mapper_hidden_dim),
                                        nn.ReLU(inplace=True),
                                    ] * self.mapper_layer,
                                    nn.Conv2d(self.mapper_hidden_dim, self.mapper_out_dim, 3,1,1),
                                    # nn.BatchNorm2d(self.mapper_out_dim),
                                    nn.ReLU(inplace=True),
                                )
            self.height_mapper.apply(weights_init)

        self.late_fusion_mapper = None
        if getattr(fuser_cfg, 'use_late_fusion_mapper', False):
            self.late_fuse_mode = fuser_cfg.late_fuse_mode
            assert self.late_fuse_mode in ['+', '*']
            self.late_mapper_in_dim = fuser_cfg.late_mapper_in_dim
            self.late_mapper_hidden_dim = fuser_cfg.late_mapper_hidden_dim
            self.late_mapper_out_dim = fuser_cfg.late_mapper_out_dim
            self.late_fusion_mapper = nn.Sequential(
                                        nn.Conv2d(self.late_mapper_in_dim, self.late_mapper_hidden_dim, 3,1,1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(self.late_mapper_hidden_dim, self.late_mapper_hidden_dim, 3,1,1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(self.late_mapper_hidden_dim, self.late_mapper_out_dim, 3,1,1),
                                    )

            self.late_fusion_mapper.apply(weights_init)


        self.uncert_inchannel = uncert_inchannel
        self.uncert_hidden = fuser_cfg.uncert_hidden
        self.uncert_out_channel = fuser_cfg.uncert_out_channel

        if fuser_cfg.encoder_type == 'simple':
            self.weight_layer = nn.Sequential(
                nn.Conv2d(self.uncert_inchannel, self.uncert_hidden,3,1,1, bias=False),
                nn.BatchNorm2d(self.uncert_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.uncert_hidden, self.uncert_hidden,3,1,1, bias=False),
                nn.BatchNorm2d(self.uncert_hidden),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer = UNet(self.uncert_inchannel, self.uncert_hidden)
            
        self.uncertainty_head = nn.Conv2d(self.uncert_hidden, 1, kernel_size=1)
        self.div_head = nn.Conv2d(self.uncert_hidden, self.uncert_out_channel, kernel_size=1)

        self.weight_layer.apply(weights_init)
        self.uncertainty_head.apply(weights_init)
        self.div_head.apply(weights_init)

    def forward(self, feats, extra_info):
        weight_input = feats.clone()
        if self.height_mapper is not None:
            sample_depth = extra_info['sample_depth'].cuda()
            nerf_depth = extra_info['nerf_depth'].cuda()
            disp = (nerf_depth-sample_depth)**2
            disp = (disp-disp.min()) / (disp.max()-disp.min()+1e-7)
            zero_mask = (sample_depth==0) | (nerf_depth==0)
            disp[zero_mask] = 1
            disp = torch.nan_to_num(disp, nan=1)
            disp = self.height_mapper(disp)
            weight_input = torch.cat([weight_input, disp], dim=1)

        uncertainty_feat = self.weight_layer(weight_input)
        weight = self.uncertainty_head(uncertainty_feat)

        if self.late_fusion_mapper is not None:
            var = extra_info['var'].cuda()
            var_weight = self.late_fusion_mapper(var)
            if self.late_fuse_mode == '*':
                weight = weight * F.sigmoid(var_weight)
            elif self.late_fuse_mode == '+':
                weight = weight + var_weight

        pred_kl_div = self.div_head(uncertainty_feat)
        pred_kl_div = F.sigmoid(pred_kl_div)
        return weight, pred_kl_div

