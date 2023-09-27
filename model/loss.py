import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.voxel_utils import reduce_masked_mean


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, weight, reduce='weight'):
        super(SoftCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.loss = nn.CrossEntropyLoss(weight)
        self.reduce = reduce

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)

class FocalLoss(nn.Module):
    def __init__(self, weight, alpha=1, gamma=2, reduce='weight'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduce = reduce

    def forward(self, inputs, targets, valid=None):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce == 'mean':
            return torch.mean(F_loss)
        elif self.reduce == 'sum':
            return torch.sum(F_loss)
        elif self.reduce == 'weight':
            F_loss = F_loss.sum(axis=1)
            return torch.mean(F_loss)
        else:
            raise NotImplementedError


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid=None):
        loss = self.loss_fn(ypred, ytgt)
        if valid is not None:
            loss = (loss*valid)
        
        return loss.mean()

class FeatLoss(torch.nn.Module):
    def __init__(self, weight):
        super(FeatLoss, self).__init__()
        self.loss_fn = torch.nn.KLDivLoss()
        self.weight = weight

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt) * self.weight
        return loss

class AuxiliaryLoss(torch.nn.Module):
    def __init__(self, weight):
        super(AuxiliaryLoss, self).__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.weight = weight

    def forward(self, pred_kl_div, semantic, semantic_gt):
        semantic = F.log_softmax(semantic)
        gt_kl_div = F.kl_div(semantic, semantic_gt, reduction='none')
        loss = self.loss_fn(pred_kl_div, gt_kl_div) * self.weight
        return loss

def build_loss_fn(cfg, device):
    loss_fn = {}
    for key in cfg.keys():
        if key == 'rgb_loss':
            loss_fn[key] = F.mse_loss
        elif key == 'depth_loss':
            loss_fn[key] = F.l1_loss
        elif key == 'seg_loss':
            if cfg[key].type == 'simple':
                loss_fn[key] = SimpleLoss(cfg.seg_loss.pose_weight).to(device)
            elif cfg[key].type == 'focal_loss':
                loss_fn[key] = FocalLoss(weight=cfg.seg_loss.weights, alpha=cfg.seg_loss.alpha, gamma=cfg.seg_loss.gamma).to(device)
            else:
                raise NotImplementedError
        elif key == 'feat_loss':
            loss_fn[key] = FeatLoss(cfg.feat_loss.weight).to(device)
        elif key == 'aux_loss':
            loss_fn[key] = AuxiliaryLoss(cfg.aux_loss.weight).to(device)
        elif key == 'soft_loss':
            loss_fn[key] = SoftCrossEntropyLoss(cfg.soft_loss.weight).to(device)
    return loss_fn