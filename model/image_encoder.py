import torch
import torch.nn as nn
import torchvision

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, size=(900,1600), scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(size=size, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class ResNet(nn.Module):
    def __init__(self, C, size, type=50, init_ckpt=None):
        super().__init__()
        self.C = C
        if type == 18:
            resnet = torchvision.models.resnet18(pretrained=True)
        elif type == 50:
            resnet = torchvision.models.resnet50(pretrained=True)
        elif type == 101:
            resnet = torchvision.models.resnet101(pretrained=True)
        else:
            raise NotImplementedError
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512, size) #UpsamplingConcat(192, 256)

        if init_ckpt:
            param = torch.load(init_ckpt)
            backbone_param = {k.replace('encoder.backbone.', ''):v for k,v in param['model_state_dict'].items() if 'encoder.backbone' in k}
            layer3_param = {k.replace('encoder.layer3.', ''):v for k,v in param['model_state_dict'].items() if 'encoder.layer3' in k}
            depth_layer_param = {k.replace('encoder.depth_layer.', ''):v for k,v in param['model_state_dict'].items() if 'encoder.depth_layer' in k}
            self.backbone.load_state_dict(backbone_param)
            self.layer3.load_state_dict(layer3_param)
            self.depth_layer.load_state_dict(depth_layer_param)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class ImgEncoder(nn.Module):
    def __init__(self, cfg, device):
        super(ImgEncoder, self).__init__()
        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().to(device)
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().to(device)
        self.encoder = ResNet(cfg.feat_dim, cfg.size, cfg.type, cfg.init_ckpt)

    def forward(self, imgs):
        imgs = (imgs + 0.5 - self.mean) / self.std
        feats = self.encoder(imgs)
        return feats