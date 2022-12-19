import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as ms_np
from mindspore import ms_function
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindelec.architecture import MultiScaleFCCell
from mindelec.operators import SecondOrderGrad, Grad
from mindelec.common import PI

import os
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.config import maxwell_3d_config

def plot_waveguide(label, pred, diff, save_dir=""):
    fig, axes =  plt.subplots(3, 1, figsize=(16, 10))
    axes[0].plot(label[:,2], 'g--', label="Ez ground truth")
    axes[0].plot(pred[:,2], 'r--', label="Ez prediction")
    axes[0].set_xlabel("points")
    axes[0].set_ylabel("value")
    axes[0].legend()
    axes[0].set_title("Ez")
    
    axes[1].plot(pred[:, 0], 'm-', label="Ex prediction")
    axes[1].plot(pred[:, 1], 'c-', label="Ex prediction")
    axes[1].set_xlabel("points")
    axes[1].set_ylabel("value")
    axes[1].legend()
    axes[1].set_title("Ex and Ey prediction")
    
    axes[2].plot(diff[:, 2], 'b-')
    axes[2].set_xlabel("points")
    axes[2].set_ylabel("value")
    axes[2].set_title("Ez difference")
    
    plt.savefig(f"{save_dir}/waveguide.png", dpi=100, bbox_inches='tight')
    # plt.show()

    
def plot_domain_plane(u1, u2, u3, xyrange, save_dir=""):
    vmax = np.max([u1, u2, u3])
    vmin = np.min([u1, u2, u3])
    fig, axes =  plt.subplots(4, 3, figsize=(10, 10))
    for e in range(3):
        ax = axes[0, e].imshow(u1[e], vmin=vmin, vmax=vmax, cmap='jet')
        axes[0, e].set_xticks([0, len(u1[e])], xyrange)
        axes[0, e].set_yticks([0, len(u1[e])], xyrange)
    for e in range(3):
        ax = axes[1, e].imshow(u2[e], vmin=vmin, vmax=vmax, cmap='jet')
        axes[1, e].set_xticks([0, len(u2[e])], xyrange)
        axes[1, e].set_yticks([0, len(u2[e])], xyrange)
    for e in range(3):
        ax = axes[2, e].imshow(u3[e], vmin=vmin, vmax=vmax, cmap='jet')
        axes[2, e].set_xticks([0, len(u3[e])], xyrange)
        axes[2, e].set_yticks([0, len(u3[e])], xyrange)
    fig.colorbar(ax, ax=[axes[e, xyz] for e in range(3)  for xyz in range(3)], shrink=0.6)
    
    for e in range(3):
        axes[3, e].set_xticks([])
        axes[3, e].set_yticks([])
        axes[3, e].spines['top'].set_visible(False)
        axes[3, e].spines['right'].set_visible(False)
        axes[3, e].spines['bottom'].set_visible(False)
        axes[3, e].spines['left'].set_visible(False)
    text = plt.text(x=-2,#文本x轴坐标 
         y=0.5, #文本y轴坐标
         s='Top to down is 3 planes: x=0, y=0, z=0\nLeft to right are 3 components: $E_x$, $E_y$, $E_z$', #文本内容
         fontdict=dict(fontsize=12, color='r',family='monospace',),#字体属性字典
         #添加文字背景色
         bbox={'facecolor': '#74C476', #填充色
              'edgecolor':'b',#外框色
               'alpha': 0.5, #框透明度
               'pad': 8,#本文与框周围距离 
              }
        )
    text.set_color('b')#修改文字颜色
    
    plt.savefig(f"{save_dir}/domain_predict.png", dpi=100, bbox_inches='tight')
    # plt.show()

    
# 定义损失函数
class TestMaxwell3DSlab():
    """定义带损失函数的网络，采用L2损失，即MSR误差"""
    def __init__(self, config):
        self.config = config
        
        self.net = self.init_net()
        self.hessian_Ex_xx = SecondOrderGrad(self.net, 0, 0, output_idx=0)
        self.hessian_Ex_yy = SecondOrderGrad(self.net, 1, 1, output_idx=0)                     
        self.hessian_Ex_zz = SecondOrderGrad(self.net, 2, 2, output_idx=0)                     
        
        self.hessian_Ey_xx = SecondOrderGrad(self.net, 0, 0, output_idx=1)
        self.hessian_Ey_yy = SecondOrderGrad(self.net, 1, 1, output_idx=1)
        self.hessian_Ey_zz = SecondOrderGrad(self.net, 2, 2, output_idx=1)
        
        self.hessian_Ez_xx = SecondOrderGrad(self.net, 0, 0, output_idx=2)
        self.hessian_Ez_yy = SecondOrderGrad(self.net, 1, 1, output_idx=2)
        self.hessian_Ez_zz = SecondOrderGrad(self.net, 2, 2, output_idx=2)
        
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(1)
        self.abs = ops.Abs()
        
        self.zeros_like = ops.ZerosLike()
        self.reduce_mean = ops.ReduceMean()
        
        self.wave_number = Tensor(self.config["wave_number"], ms.dtype.float32) # 波数
        self.pi = Tensor(PI, ms.dtype.float32) # 常数pi
        
        # 坐标轴范围
        self.xyrange = (self.config["coord_min"][0], self.config["coord_max"][0]) 
    
    def init_net(self):
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
        
        net = MultiScaleFCCell(in_channel=self.config["in_channel"], 
                   out_channel=self.config["out_channel"], 
                   layers=self.config["layers"],
                   neurons=self.config["neurons"],
                  )
        load_paramters_into_net(self.config["param_path"],  net)
        return net
        

    def run(self):
        """构造数据，检查输出结果"""
        print("<===================== Begin evaluating =====================>")
        t_start = time.time()
        xmin, ymin, zmin = self.config["coord_min"]
        xmax, ymax, zmax = self.config["coord_max"]
        xyrange = (xmin, xmax)
        axis_size = self.config["axis_size"]
        save_dir = self.config["result_save_dir"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_dir.endswith('/'):
            save_dir = save_dir[:-1]
    
        u = np.linspace(xmin, xmax, axis_size) # 认为所选区域是个正方体
        v = np.linspace(ymin, ymax, axis_size)
        U, V = np.meshgrid(u, v)
        uu, vv = U.reshape(-1,1), V.reshape(-1,1)
        ones = np.ones_like(uu)

        # 检测的4个平面
        plane1 = np.c_[ones*0, uu, vv] # x=0, yz, 
        plane2 = np.c_[uu, ones*0, vv] # xz, y=0
        plane3 = np.c_[uu, vv, ones*0] # xy, z=0
    
        # 平面0，波导管平面
        label0, E0, diff0 = self.get_waveguide_residual(self.config["waveguide_points_path"])
        print(f"Max difference of waveguide port in Ex: {diff0[:, 0].max():.5f}")
        print(f"Max difference of waveguide port in Ey: {diff0[:, 1].max():.5f}")
        print(f"Max difference of waveguide port in Ez: {diff0[:, 2].max():.5f}")
        plot_waveguide(label0, E0, diff0, save_dir)
        print("plot waveguide completed!")
        
        # 平面1，2，3
        E1 = self.net(ms.Tensor(plane1, ms.dtype.float32)).asnumpy()  # x=0, yz平面
        E2 = self.net(ms.Tensor(plane2, ms.dtype.float32)).asnumpy()  # x=0, yz平面
        E3 = self.net(ms.Tensor(plane3, ms.dtype.float32)).asnumpy()  # x=0, yz平面
        
        E1 = E1.reshape((U.shape[0], U.shape[1], 3)).transpose(2, 0, 1)
        E2 = E2.reshape((U.shape[0], U.shape[1], 3)).transpose(2, 0, 1)
        E3 = E3.reshape((U.shape[0], U.shape[1], 3)).transpose(2, 0, 1)
        plot_domain_plane(E1, E2, E3, xyrange, save_dir)
        print("plot domain result completed!")
        
        
        # 对整个体空间计算结果
        print("Begin scan the whole volumn, it may take a long time.")
        # result[i, x, y, z]
        # i=0 -> Ex,  i=1 -> Ey, i=2 -> Ez
        # (x,y,z)为对应点坐标
        result = np.zeros(shape=(3, axis_size, axis_size, axis_size), dtype=np.float32)
        for i, x in tqdm.tqdm(enumerate(np.linspace(xmin, xmax, axis_size))):
            xx = ones * x
            points = ms.Tensor(np.c_[xx, uu, vv], ms.dtype.float32)
            u_xyz = self.net(points).asnumpy()
            result[0, i, :, :] = u_xyz[:, 0].reshape((axis_size, axis_size))
            result[1, i, :, :] = u_xyz[:, 1].reshape((axis_size, axis_size))
            result[2, i, :, :] = u_xyz[:, 2].reshape((axis_size, axis_size))
        np.save(f"{save_dir}/slab_result.npy", result)
        
        print("<===================== End evaluating =====================>")
        t_end = time.time()
        print(f"This evaluation total spend {(t_end - t_start) / 60:.2f} minutes.")
    
    
    def get_waveguide_residual(self, data_path):
        """根据边界条件计算边界损失，包括左面的波导面约束
         Args:
             data: shape=(n,3), n为点数，3列分别代表x,y,z坐标
        
        Return:
            label: shape=(n,3), 真实值
            u: shape=(n,3), 3列分别代表输入点对应的Ex, Ey, Ez
            diff: shape=(n,3), 真实值label和预测值u的差值
        """
        data = np.load(data_path) # shape=(13277, 9)
        net_inputs = ms.Tensor(data[:, 0:3], ms.dtype.float32)
        net_outputs = self.net(net_inputs)
        
        pred = net_outputs.asnumpy()
        ## Ez = sin(m * pi * y / height) * sin(m * pi * y / length)
        label_z = data[:, 3:4] # 第4-9列均可为label,训练中使用第4列
        label = np.c_[np.zeros_like(label_z), np.zeros_like(label_z), label_z]
        diff = np.abs(pred - label)
        return label, pred, diff
    
    
if __name__ == "__main__":
    tester = TestMaxwell3DSlab(config=maxwell_3d_config)
    tester.run()
    
    """
    # 可使用如下代码查看运行结束后保存的 xxx.npy 中整个立方体的Ex, Ey, Ez结果
    # result[i, x, y, z] 代表第i个通道在点(x,y,z)的值
    # x,y,z取值均为 [0, axis_size-1]，默认 axis_size=101
    # i=0表示Ex, i=1表示Ey, i=2表示Ez
    # 若要查看整个平面的值，对应下标用冒号代替
    
    import numpy as np
    import matplotlib.pyplot as plt
    result = np.load('result/slab_result.npy')
    plt.imshow(result[2, :, :, 51], cmap='jet')
    plt.colorbar()
    plt.show()
    """
    
    