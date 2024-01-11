# ===================================================================================
# 相关的配置数据
# ===================================================================================
class CFG:
    # dataset相关
    # data_root = "/home/magic/magic/Datasets/modelnet/modelnet40_normal_resampled"  # w7
    data_root = "/home/magic2/Datasets/modelnet/modelnet40_normal_resampled"  # w4
    class_name = "chair"
    voxel_num = 32  # 每个坐标轴体素的数量

    # 训练相关
    batch_size = 48
    num_workers = 8
    epoch = 100
    voxel_dim = 32
    noise_dim = 32

    ckpt_dir = "./ckpt"
    save_best = True

    # 生成相关
    ckpt_path = "./ckpt/0best_model.pth"
    gene_num = 2
    gene_step = 20
    result_dir = "./generation"