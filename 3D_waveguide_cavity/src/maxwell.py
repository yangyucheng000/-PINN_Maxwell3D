import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as ms_np
from mindspore import ms_function
from mindspore import Tensor

from mindelec.operators import SecondOrderGrad, Grad
from mindelec.common import PI


# 定义损失函数
class Maxwell3DLoss(nn.Cell):
    """定义带损失函数的网络，采用L2损失，即MSR误差"""
    def __init__(self, net, config):
        super(Maxwell3DLoss, self).__init__(auto_prefix=False)
        self.net = net
        self.config = config
        
        self.grad = Grad(net)  # 梯度算子
        
        self.hessian_Ex_xx = SecondOrderGrad(net, 0, 0, output_idx=0)
        self.hessian_Ex_yy = SecondOrderGrad(net, 1, 1, output_idx=0)                     
        self.hessian_Ex_zz = SecondOrderGrad(net, 2, 2, output_idx=0)                     
        
        self.hessian_Ey_xx = SecondOrderGrad(net, 0, 0, output_idx=1)
        self.hessian_Ey_yy = SecondOrderGrad(net, 1, 1, output_idx=1)
        self.hessian_Ey_zz = SecondOrderGrad(net, 2, 2, output_idx=1)
        
        self.hessian_Ez_xx = SecondOrderGrad(net, 0, 0, output_idx=2)
        self.hessian_Ez_yy = SecondOrderGrad(net, 1, 1, output_idx=2)
        self.hessian_Ez_zz = SecondOrderGrad(net, 2, 2, output_idx=2)
        
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(1)
        self.cast = ops.Cast()
        self.mul = ops.Mul()
        
        self.zeros_like = ops.ZerosLike()
        self.reduce_mean = ops.ReduceMean()
        self.l2_loss = nn.MSELoss() # 损失函数
        
        self.eps0 = Tensor(config["eps0"], ms.dtype.float32) # 电导率系数
        self.wave_number = Tensor(config["wave_number"], ms.dtype.float32) # 波数
        self.pi = Tensor(PI, ms.dtype.float32) # 常数pi
        self.eigenmode = Tensor(config["eigenmode"], ms.dtype.float32) # 特征模数
    
        self.gamma_domain = Tensor(1.0 / self.wave_number**2, ms.dtype.float32) # 立方体中心采样点损失加权系数
        self.gamma_bc = Tensor(100.0, ms.dtype.float32)  # 6个表面的损失加权系数


    def construct(self, in_domain, in_bc):
        """输入数据，计算损失"""
        # 计算内部损失
        u_domain = self.net(in_domain)
        loss_domain = self.get_domain_loss(u_domain, in_domain)
        
        # 计算边界损失
        u_bc = self.net(in_bc)
        loss_bc = self.get_boundary_loss(u_bc, in_bc)
        
        return self.gamma_domain * loss_domain + self.gamma_bc * loss_bc
    
    
    @ms_function    
    def get_domain_loss(self, u, data):
        """根据3D Maxwell方程，计算采样点内部损失"""
        # 求二阶导数
        Ex_xx = self.hessian_Ex_xx(data)        
        Ex_yy = self.hessian_Ex_yy(data)        
        Ex_zz = self.hessian_Ex_zz(data)
        
        Ey_xx = self.hessian_Ey_xx(data)
        Ey_yy = self.hessian_Ey_yy(data)
        Ey_zz = self.hessian_Ey_zz(data)
        
        Ez_xx = self.hessian_Ez_xx(data)
        Ez_yy = self.hessian_Ez_yy(data)
        Ez_zz = self.hessian_Ez_zz(data)

        # 分三项计算pde方程的残差(residual)值
        pde_rx = Ex_xx + Ex_yy + Ex_zz + self.wave_number**2 * self.eps0 * u[:, 0:1]
        pde_ry = Ey_xx + Ey_yy + Ey_zz + self.wave_number**2 * self.eps0 * u[:, 1:2]
        pde_rz = Ez_xx + Ez_yy + Ez_zz + self.wave_number**2 * self.eps0 * u[:, 2:3]
        
        pde_r = self.concat((pde_rx, pde_ry, pde_rz))
        
        # 方程右边为0，与0计算L2损失
        loss_domain = self.reduce_mean(self.l2_loss(pde_r, self.zeros_like(pde_r)))
        
        # 第二项损失
        Ex_x = self.grad(data, 0, 0, u) 
        Ey_y = self.grad(data, 1, 1, u)
        Ez_z = self.grad(data, 2, 2, u)
        no_source_r = Ex_x + Ey_y + Ez_z  # 无源，散度为0
        
        loss_no_source = self.reduce_mean(self.l2_loss(no_source_r, self.zeros_like(no_source_r)))
        
        return loss_domain + loss_no_source
    
    
    @ms_function
    def get_boundary_loss(self, u, data):
        """根据边界条件计算边界损失，包括左面的波导面约束"""
        coord_min = self.config["coord_min"]
        coord_max = self.config["coord_max"]
        batch_size, _ = data.shape
        # mask用于选择部分数据
        mask = ms_np.zeros(shape=(batch_size, 17), dtype=ms.dtype.float32)
        
        mask[:, 0] = ms_np.where(ms_np.isclose(data[:, 0], coord_min[0]), 1.0, 0.0) # left 
        mask[:, 1] = ms_np.where(ms_np.isclose(data[:, 0], coord_min[0]), 1.0, 0.0) # left 
        mask[:, 2] = ms_np.where(ms_np.isclose(data[:, 0], coord_min[0]), 1.0, 0.0) # left 
        
        mask[:, 3] = ms_np.where(ms_np.isclose(data[:, 0], coord_max[0]), 1.0, 0.0) # right
        mask[:, 4] = ms_np.where(ms_np.isclose(data[:, 0], coord_max[0]), 1.0, 0.0) # right

        mask[:, 5] = ms_np.where(ms_np.isclose(data[:, 1], coord_min[1]), 1.0, 0.0) # bottom        
        mask[:, 6] = ms_np.where(ms_np.isclose(data[:, 1], coord_min[1]), 1.0, 0.0) # bottom
        mask[:, 7] = ms_np.where(ms_np.isclose(data[:, 1], coord_min[1]), 1.0, 0.0) # bottom

        mask[:, 8] = ms_np.where(ms_np.isclose(data[:, 1], coord_max[1]), 1.0, 0.0) # top
        mask[:, 9] = ms_np.where(ms_np.isclose(data[:, 1], coord_max[1]), 1.0, 0.0) # top
        mask[:, 10] = ms_np.where(ms_np.isclose(data[:, 1], coord_max[1]), 1.0, 0.0) # top

        mask[:, 11] = ms_np.where(ms_np.isclose(data[:, 2], coord_min[2]), 1.0, 0.0) # back
        mask[:, 12] = ms_np.where(ms_np.isclose(data[:, 2], coord_min[2]), 1.0, 0.0) # back
        mask[:, 13] = ms_np.where(ms_np.isclose(data[:, 2], coord_min[2]), 1.0, 0.0) # back

        mask[:, 14] = ms_np.where(ms_np.isclose(data[:, 2], coord_max[2]), 1.0, 0.0) # front
        mask[:, 15] = ms_np.where(ms_np.isclose(data[:, 2], coord_max[2]), 1.0, 0.0) # front
        mask[:, 16] = ms_np.where(ms_np.isclose(data[:, 2], coord_max[2]), 1.0, 0.0) # front
        
        # 左面 Waveguide-port
        ## Ex = sin(m * pi * y / height) * sin(m * pi * y / length)
        height = coord_max[1] # y轴最大值为高度，y轴朝上
        length = coord_max[2] # z轴最大值为长度，z轴朝外
        # data[:,0]->x, data[:,1]->y, data[:,2]->z
        label_left = ops.sin(self.eigenmode * self.pi * data[:, 1:2] / height) * \
                        ops.sin(self.eigenmode * self.pi * data[:, 2:3] / length)
        bc_r_waveguide = 10.0 * (u[:, 2:3] - label_left) # z方向分布，相对其他边界赋予更大损失，保证边界满足要求
        bc_r_left = self.concat([u[:, 0:1], u[:, 1:2], bc_r_waveguide]) # 左面的Ex, Ey为0

        # 右面的ABC损失
        ## n x \nabla x E = 0 ==> dEz/dy - dEy/dz = 0
        # Ez_y = self.grad(data, 1, 2, u)
        # Ey_z = self.grad(data, 2, 1, u)
        # bc_r_right = Ez_y - Ey_z
        
        Ex_z = self.grad(data, 2, 0, u)
        Ez_x = self.grad(data, 0, 2, u)
        
        Ey_x = self.grad(data, 0, 1, u)
        Ex_y = self.grad(data, 1, 0, u)
        
        bc_r_right = self.concat([Ex_z - Ez_x, Ey_x - Ex_y])
        

        # 四面的PEC损失 n x E = 0
        ## top: Ez, Ex = 0
        ## bottom: Ez, Ex = 0
        ## front: Ex, Ey = 0
        ## back: Ex, Ey = 0
        Ey_y = self.grad(data, 1, 1, u)
        Ez_z = self.grad(data, 2, 2, u)
        bc_r_round = self.concat(
            (u[:, 2:3], u[:, 0:1], Ey_y,  ## bottom: Ez=0, Ex = 0, dEy/dy=0
             u[:, 2:3], u[:, 0:1], Ey_y,  ## top: Ez, Ex = 0, dEy/dy=0
             u[:, 0:1], u[:, 1:2], Ez_z,  ## back: Ex, Ey = 0, dEz/dz=0
             u[:, 0:1], u[:, 1:2], Ez_z), ## front: Ex, Ey = 0, dEz/dz=0
        )

        bc_r_all = self.concat((bc_r_left, bc_r_right, bc_r_round))
        bc_r = self.mul(bc_r_all, mask)
        
        # 方程右边为0，与0计算L2损失
        loss_bc = self.reduce_mean(self.l2_loss(bc_r, self.zeros_like(bc_r)))
        return loss_bc
        
