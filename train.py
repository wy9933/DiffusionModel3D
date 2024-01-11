import tqdm
import os
import open3d as o3d

import torch.utils.data as data
import torch.nn.functional as F

from dataset3d import Dataset3D
from dm3d import DiffusionModel3D
from utils import *
from cfg import CFG


def main():
    # 定义相关配置
    cfg = CFG()

    # 定义dataloader
    dataset = Dataset3D(cfg.data_root, cfg.class_name, cfg.voxel_num)
    dataloader = data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    # 定义模型
    model = DiffusionModel3D(cfg.voxel_dim, cfg.noise_dim).cuda()

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    # 定义损失函数
    loss_func = F.mse_loss

    train(model, dataloader, optimizer, loss_func, cfg)


def train(model, dataloader, optimizer, loss_func, cfg):
    model.train()

    best_loss = 99999999
    for i in range(cfg.epoch):
        epoch_loss = 0
        for b, voxels in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            batch_size = voxels.shape[0]

            # 归一化输入数据
            voxels = F.normalize(voxels, dim=0).cuda()
            # 生成随机噪声
            noises = torch.randn(voxels.shape).cuda()
            # 生成随机时间步来生成数据和噪声的占比，然后叠加得到加噪数据
            times = torch.rand((batch_size, 1, 1, 1, 1)).cuda()
            noise_rates, signal_rates = offset_cosine_diffusion_schedule_torch(times)
            noise_voxels = signal_rates * voxels + noise_rates * noises

            # 预测噪声
            pred_noises = model(noise_voxels, noise_rates**2)

            # 计算损失并传播梯度
            loss = loss_func(noises, pred_noises)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        print("Epoch {}, loss: {}".format(i, epoch_loss))

        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        filename = os.path.join(cfg.ckpt_dir, 'last_model.pth')
        torch.save({
            'epoch': i,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename)
        if i % 10 == 0:
            filename = os.path.join(cfg.ckpt_dir, 'epoch_%d_model.pth' % i)
            torch.save({
                'epoch': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename)
        if cfg.save_best and epoch_loss < best_loss:
            best_loss = epoch_loss
            filename = os.path.join(cfg.ckpt_dir, 'best_model.pth')
            torch.save({
                'epoch': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename)


if __name__ == "__main__":
    main()