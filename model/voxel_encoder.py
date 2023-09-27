import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
  def __init__(self, idims=64, dims=64, num_layers=1,
               stride=1):
    super(EncoderBlock, self).__init__()
    layers = []
    self.idims = idims
    self.stride = stride
    for i in range(num_layers):
      layers.append(nn.Conv2d(self.idims, dims, 3, stride=self.stride,
                              padding=1, bias=False))
      layers.append(nn.InstanceNorm2d(dims))
      layers.append(nn.GELU())
      self.idims = dims
      self.stride = 1
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    return self.layers(x)

class VoxelEncoder(nn.Module):
  def __init__(self, encoder_cfg):
    super(VoxelEncoder, self).__init__()
    self.encoder_cfg = encoder_cfg
    in_channel = encoder_cfg.in_channel 

    self.voxel_encoder = nn.Sequential(
        nn.Conv2d(in_channel, 128, kernel_size=3, padding=1, stride=1, bias=False),
        nn.InstanceNorm2d(128),
        nn.GELU(),
    )
    self.block1 = EncoderBlock(128, dims=128, num_layers=2, stride=2)
    self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.block2 = EncoderBlock(128, dims=256, num_layers=2, stride=4)
    self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    out_channel = encoder_cfg.out_channel 
    self.conv_out = nn.Sequential(
      nn.Conv2d(512, out_channel, kernel_size=3, padding=1),
      nn.InstanceNorm2d(out_channel),
      nn.GELU(),
    )

  def forward(self, voxels):
    voxel_feature = self.voxel_encoder(voxels)
    voxel_feature1 = self.block1(voxel_feature)
    voxel_feature1 = self.up1(voxel_feature1)
    voxel_feature2 = self.block2(voxel_feature)
    voxel_feature2 = self.up2(voxel_feature2)
    bev_feature = torch.cat([voxel_feature, voxel_feature1, voxel_feature2], dim=1)
    return self.conv_out(bev_feature)


