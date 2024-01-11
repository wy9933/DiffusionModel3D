import numpy as np
import torch
import math


# ===================================================================================
# 生成加入偏置的余弦扩散计划
# ===================================================================================
def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = np.arccos(max_signal_rate)
    end_angle = np.arccos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = np.cos(diffusion_angles)
    noise_rates = np.sin(diffusion_angles)
    signal_rates = torch.FloatTensor(signal_rates)
    noise_rates = torch.FloatTensor(noise_rates)

    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule_torch(diffusion_times):
    min_signal_rate = torch.tensor(0.02).to(diffusion_times.device)
    max_signal_rate = torch.tensor(0.95).to(diffusion_times.device)
    start_angle = torch.acos(max_signal_rate)
    end_angle = torch.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)

    return noise_rates, signal_rates


# ===================================================================================
# 将单一噪声方差标量值转化为一个长为 dim 的向量
# ===================================================================================
def sinusoidal_embedding(x, dim):
    frequencies = np.exp(np.linspace(np.math.log(1.0), np.math.log(1000.0), int(dim / 2)))
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = np.concatenate([np.sin(angular_speeds * x), np.cos(angular_speeds * x)], axis=-1)
    embeddings = torch.FloatTensor(embeddings)

    return embeddings


def sinusoidal_embedding_torch(x, dim):
    frequencies = torch.exp(torch.linspace(np.math.log(1.0), np.math.log(1000.0), int(dim / 2))).to(x.device)
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = torch.cat([torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)], dim=-1)

    return embeddings
