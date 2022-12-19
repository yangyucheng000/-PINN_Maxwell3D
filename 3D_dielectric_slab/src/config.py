from easydict import EasyDict as ed


# 在立方体中采样配置
cuboid_sampling_config = ed({
    'domain': ed({                   # 区域内采样
        'random_sampling': False,    # 是否随机采样
        'size': [64, 64, 64],        # 采样网格
    }),
    'BC': ed({                       # 边界点采样
        'random_sampling': True,     # 是否随机采样
        'size': 65536,               # 采样点数
        'sampler': 'uniform',        # 均匀分布采样
    })
})


# 模型等config
maxwell_3d_config = ed({
    "name": "Maxwell3D",             # 模型名称
    "geom_name": "cuboid",           # 几何体名称
    "waveguide_name": "waveguide",   # 波导管名称
    "waveguide_points_path": "data/sample_points_all.npy", # 波导管数据
    
    # 训练参数
    "epochs": 3000,                  # 迭代epoch次数，建议 >=1000
    "batch_size": 256,               # 训练batch_size
    "lr": 0.001,                     # 训练学习率
    "pretrained": False,             # 是否使用预训练模型，模型从 param_path加载。
    "param_path": "checkpoints/model_slab_best.ckpt",  # 训练好的参数路径
    
    # 仿真环境参数
    "coord_min": [-0.5, -0.5, -0.5], # xyz最小坐标值
    "coord_max": [0.5, 0.5, 0.5],    # xyz最大坐标值
    "slab_len": 0.2,                 # 中间电导体尺寸
    "eps1": 1.5,                     # 电导体电导率
    "eps0": 1.0,                     # 真空电导率
    "wave_number": 32.0,             # 波数
    
    # 神经网络参数
    "in_channel": 3,                 # 输入通道数
    "out_channel": 3,                # 输出通道数
    "layers": 6,                     # 重复次数
    "neurons": 32,                   # 每层的神经元数量
    
    # 评估参数，其中训练好权重路径和训练参数param_path共用
    "axis_size": 101,                # 评估时网格划分密度
    "result_save_dir": "result",     # 评估结果保存文件夹
})
