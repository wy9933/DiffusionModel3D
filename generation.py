import tqdm
import os
import open3d as o3d

import torch.utils.data as data
import torch.nn.functional as F

from dataset3d import Dataset3D
from dm3d import DiffusionModel3D
from utils import *
from cfg import CFG


def generation():
    # 定义相关配置
    cfg = CFG()

    # 定义模型
    model = DiffusionModel3D(cfg.voxel_dim, cfg.noise_dim).cuda()
    checkpoint = torch.load(cfg.ckpt_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    initial_noises = torch.randn((cfg.gene_num, cfg.voxel_num, cfg.voxel_num, cfg.voxel_num, 1)).cuda()
    step_size = 1.0 / cfg.gene_step
    current_voxels = initial_noises
    for step in tqdm.tqdm(range(cfg.gene_step)):
        # 计算当前step对应的扩散计划
        times = torch.ones((cfg.gene_num, 1, 1, 1, 1)).cuda() - step * step_size
        noise_rates, signal_rates = offset_cosine_diffusion_schedule_torch(times)

        # 预测噪声，并计算x0的近似值
        pred_noises = model(current_voxels, noise_rates**2)
        pred_voxels = (current_voxels - noise_rates * pred_noises) / signal_rates

        # 计算t-1步的加噪数据
        next_times = times - step_size
        next_noise_rates, next_signal_rates = offset_cosine_diffusion_schedule_torch(next_times)
        current_voxels = next_signal_rates * pred_voxels + next_noise_rates * pred_noises

        torch.cuda.empty_cache()

    voxels = pred_voxels.detach().cpu().numpy()
    os.makedirs(cfg.result_dir, exist_ok=True)
    for i, voxel in enumerate(voxels):
        voxel -= np.min(voxel)
        voxel /= np.max(voxel)
        voxel = voxel.reshape(cfg.voxel_num, cfg.voxel_num, cfg.voxel_num)
        voxel = np.where(voxel < 0.5, 0, 1)
        voxel = np.argwhere(voxel == 1).astype(float) / cfg.voxel_num

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel)
        o3d.io.write_point_cloud(os.path.join(cfg.result_dir, "{}.ply".format(i)), pcd)


if __name__ == "__main__":
    generation()