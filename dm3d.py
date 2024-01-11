import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        if input_dim != output_dim:
            self.conv_r = nn.Conv3d(input_dim, output_dim, kernel_size=1, stride=1)
        else:
            self.conv_r = nn.Identity()

        self.bn = nn.BatchNorm3d(input_dim)
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.conv2 = nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv_r(x)
        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += r
        return x


class DownBlock(nn.Module):
    def __init__(self, input_dim, output_dim, block_num=2, stride=2):
        super().__init__()

        self.rbs = nn.Sequential()
        last_input_dim = input_dim
        for i in range(block_num):
            self.rbs.append(ResidualBlock(last_input_dim, output_dim))
            last_input_dim = output_dim
        self.pool = nn.AvgPool3d(stride, stride)

    def forward(self, x):
        mid_res = []
        for rb in self.rbs:
            x = rb(x)
            mid_res.append(x)
        x = self.pool(x)
        return x, mid_res


class UpBlock(nn.Module):
    def __init__(self, input_dims, output_dim, block_num=2, scale=2):
        super().__init__()

        self.scale = scale

        self.rbs = nn.Sequential()
        for i in range(block_num):
            self.rbs.append(ResidualBlock(input_dims[i], output_dim))

    def forward(self, x, xd):
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear')
        for i, rb in enumerate(self.rbs):
            x = torch.cat([x, xd[i]], dim=1)
            x = rb(x)
        return x

class DiffusionModel3D(nn.Module):
    def __init__(self, voxel_dim=32, noise_dim=32):
        super().__init__()

        self.noise_dim = noise_dim

        self.conv1 = nn.Conv3d(1, voxel_dim, kernel_size=1, stride=1, padding=0)

        self.downblock1 = DownBlock(voxel_dim + noise_dim, 32)
        self.downblock2 = DownBlock(32, 64)
        self.downblock3 = DownBlock(64, 96)

        self.rb1 = ResidualBlock(96, 128)
        self.rb2 = ResidualBlock(128, 128)

        self.upblock3 = UpBlock([128 + 96, 96 + 96], 96)
        self.upblock2 = UpBlock([96 + 64, 64 + 64], 64)
        self.upblock1 = UpBlock([64 + 32, 32 + 32], 32)

        self.conv2 = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, voxel, noise):
        B, L, W, H, C = voxel.shape  # Batch, Length, Width, Height, Channel

        # Conv3D将初始体素编码为32维向量
        x = voxel.permute(0, 4, 1, 2, 3)    # B, 1, L, W, H
        x = self.conv1(x)                   # B, 32, L, W, H
        # 将noise进行位置编码并插值到和体素相同的维度
        noise_embedding = sinusoidal_embedding_torch(noise, self.noise_dim)                 # B, 1, 1, 1, 32
        noise_embedding = noise_embedding.permute(0, 4, 1, 2, 3)                            # B, 32, 1, 1, 1
        noise_embedding = F.interpolate(noise_embedding, size=(L, W, H), mode='nearest')    # B, 32, L, W, H
        # 拼接体素编码和noise编码
        x = torch.cat([x, noise_embedding], dim=1)  # B, 64, L, W, H

        x, mid_res1 = self.downblock1(x)
        x, mid_res2 = self.downblock2(x)
        x, mid_res3 = self.downblock3(x)

        x = self.rb1(x)
        x = self.rb2(x)

        x = self.upblock3(x, mid_res3)
        x = self.upblock2(x, mid_res2)
        x = self.upblock1(x, mid_res1)

        x = self.conv2(x)
        x = x.permute(0, 2, 3, 4, 1)  # B, L, W, H, 1

        return x


if __name__ == "__main__":
    dm = DiffusionModel3D().cuda()
    v = torch.randn((2, 32, 32, 32, 1)).cuda()
    n = torch.randn((2, 1, 1, 1, 1)).cuda()
    x = dm(v, n)
    print(x.shape)