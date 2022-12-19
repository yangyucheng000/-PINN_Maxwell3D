import os
import numpy as np

from mindspore import context
import mindspore as ms
import mindspore.nn as nn
from mindspore import  Model, Tensor
from mindspore.train.callback import LossMonitor
from mindspore.common.initializer import HeUniform
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindelec.architecture import MultiScaleFCCell

from src.dataset import create_train_dataset
from src.maxwell import Maxwell3DLoss
from src.config import maxwell_3d_config
from src.callback import TimeMonitor, SaveCkptMonitor

context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend")



def load_paramters_into_net(param_path, net):
    """载入训练好的参数"""
    param_dict = load_checkpoint(param_path)
    convert_ckpt_dict = {}
    for _, param in net.parameters_and_names():
        convert_name1 = "jac2.model.model.cell_list." + param.name
        convert_name2 = "jac2.model.model.cell_list." + ".".join(param.name.split(".")[2:])
        for key in [convert_name1, convert_name2]:
            if key in param_dict:
                convert_ckpt_dict[param.name] = param_dict[key]
    load_param_into_net(net, convert_ckpt_dict)
    print("Load parameters finished!")
    

def train():
    # 定义数据集
    train_dataset = create_train_dataset()
    train_loader = train_dataset.create_dataset(batch_size=maxwell_3d_config["batch_size"],
                                               shuffle=True, drop_remainder=True)
    
    # 定义可训练参数的网络
    net = MultiScaleFCCell(in_channel=maxwell_3d_config["in_channel"], 
                           out_channel=maxwell_3d_config["out_channel"], 
                           layers=maxwell_3d_config["layers"],
                           neurons=maxwell_3d_config["neurons"],
                          )
    net.to_float(ms.dtype.float16)
    
    # 是否加载预训练模型
    if maxwell_3d_config["pretrained"]:
        load_paramters_into_net(maxwell_3d_config["param_path"], net)
        
    # 定义优化器
    opt = nn.Adam(net.trainable_params(), maxwell_3d_config["lr"])
    
    # 定义损失函数网络，使用L2损失
    net_with_criterion = Maxwell3DLoss(net, maxwell_3d_config) 
    
    # 定义模型
    model = Model(net_with_criterion, loss_fn=None, optimizer=opt)
    
    # 定义Callback函数
    time_cb = TimeMonitor()
    save_cb = SaveCkptMonitor(comment="cavity") # 每n个epoch保存一次
    loss_cb = LossMonitor()
    
    # 开始训练
    model.train(maxwell_3d_config["epochs"],
                train_loader, 
                callbacks=[loss_cb, time_cb, save_cb])

    
if __name__ == "__main__":
    train()
    