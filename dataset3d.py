import os

import numpy as np
import torch
import torch.utils.data as data


# ===================================================================================
# 加载modelnet40某个类别的数据，以体素的形式输出
# ===================================================================================
class Dataset3D(data.Dataset):
    def __init__(self, data_root, class_name, voxel_num):
        super().__init__()

        self.voxel_num = voxel_num

        data_dir = os.path.join(data_root, class_name)
        self.data_files = os.listdir(data_dir)
        self.data_files = sorted(self.data_files)
        self.data_files = [os.path.join(data_dir, x) for x in self.data_files]

        self.scale = 1.1
        self.voxel_size = 2.0 * self.scale / voxel_num  # 因为有的模型坐标范围不是完美的[-1, 1]，所以放大一下体素包含的整体范围

    def __len__(self):
        return len(self.data_files)

    def voxelize(self, coord):
        """
        将空间体素化为边长voxel_num的正方体，然后根据坐标除法取整将存在点的体素置1
        :param coord: 点云坐标
        :return: 转换为voxel_num^3的体素表示，每个体素是否被占用
        """
        index = coord / self.voxel_size
        index = np.floor(index) + self.voxel_num / 2
        index = index.astype(int)
        index = np.unique(index, axis=0)
        voxel = np.zeros((self.voxel_num, self.voxel_num, self.voxel_num))
        # 这里折腾了快一小时，不能直接用一个二维数组作为下标，这样会导致直接对行进行操作
        # 竟然是需要先转置然后tuple才行
        voxel[tuple(index.T)] = 1
        return voxel

    def __getitem__(self, index):
        file_path = self.data_files[index]
        data = np.loadtxt(file_path, delimiter=',')
        coord = data[:, :3]
        voxel = self.voxelize(coord)
        voxel = torch.FloatTensor(voxel)
        voxel = voxel.unsqueeze(-1)
        return voxel