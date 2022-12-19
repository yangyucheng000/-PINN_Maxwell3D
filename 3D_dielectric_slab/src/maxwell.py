import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as ms_np
from mindspore import ms_function
from mindspore import Tensor

from mindelec.operators import SecondOrderGrad, Grad
from mindelec.common import PI


# 定义损失函数
class MaxwellCavity(nn.Cell):
    """带损失的的网络"""
    def __init__(self, net, config):
        super(MaxwellCavity, self).__init__(auto_prefix=False)
        self.net = net
        self.config = config
        
        self.grad = Grad(net)
        
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
        # self.square = ops.Square()
        
        self.zeros_like = ops.ZerosLike()
        self.reduce_mean = ops.ReduceMean()
        self.l2_loss = nn.MSELoss()
        
        self.slab_len = Tensor(config["slab_len"], ms.dtype.float32)
        self.eps1 = Tensor(config["eps1"], ms.dtype.float32)
        self.eps0 = Tensor(config["eps0"], ms.dtype.float32)
        
        self.wave_number = Tensor(config["wave_number"], ms.dtype.float32)
        self.perm = Tensor(1.0, ms.dtype.float32) # permitivity, 电容率
        self.pi = Tensor(PI, ms.dtype.float32)
        
        self.gamma_domain = Tensor(1.0/ self.wave_number**2, ms.dtype.float32)
        self.gamma_bc = Tensor(100.0, ms.dtype.float32)  # 四周PEC损失加权系数
        self.gamma_port = Tensor(100.0, ms.dtype.float32) # 左边界waveguide损失加权系数


    def construct(self, in_domain, in_bc, in_port):
        """输入数据，计算损失"""
        u_domain = self.net(in_domain)
        loss_domain = self.get_domain_loss(u_domain, in_domain)
        
        u_bc = self.net(in_bc)
        loss_bc = self.get_boundary_loss(u_bc, in_bc)
        
        u_port = self.net(in_port[:,0:3])
        loss_port = self.get_port_loss(u_port, in_port)
        
        return self.gamma_domain * loss_domain + self.gamma_bc * loss_bc \
                + self.gamma_port * loss_port

    
    @ms_function    
    def get_domain_loss(self, u, data):
        """内部损失"""
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
        
        batch_size, _ = data.shape
        mask = ms_np.zeros(shape=(batch_size, 3), dtype=ms.dtype.float32)
        # 中心slab部分为1.0
        mask = ms_np.where(data[:, 1] > -self.slab_len/2, 1.0, 0)
        mask += ms_np.where(data[:, 1] < self.slab_len/2, 1.0, 0)
        mask += ms_np.where(data[:, 2] > -self.slab_len/2, 1.0, 0)
        mask += ms_np.where(data[:, 2] < self.slab_len/2, 1.0, 0)
        # mask.shape=(batch_size,)
        # 如果在slab内，则上面四个条件均满足，此时mask对应点为1.0
        mask = ms_np.where(ms_np.isclose(mask, 4.0), 1.0, 0)
        mask = self.reshape(mask, (batch_size, 1)) # 转换成1列
        
        # slab中，介电常数取 eps1
        pde_rx_slab = Ex_xx + Ex_yy + Ex_zz + self.wave_number**2 * self.eps1 * u[:, 0:1]
        pde_ry_slab = Ey_xx + Ey_yy + Ey_zz + self.wave_number**2 * self.eps1 * u[:, 1:2]
        pde_rz_slab = Ez_xx + Ez_yy + Ez_zz + self.wave_number**2 * self.eps1 * u[:, 2:3]
        
        # 真空中，介电常数取 eps0
        pde_rx_vac = Ex_xx + Ex_yy + Ex_zz + self.wave_number**2 * self.eps0 * u[:, 0:1]
        pde_ry_vac = Ey_xx + Ey_yy + Ey_zz + self.wave_number**2 * self.eps0 * u[:, 1:2]
        pde_rz_vac = Ez_xx + Ez_yy + Ez_zz + self.wave_number**2 * self.eps0 * u[:, 2:3]
        
        pde_r_slab = self.concat((pde_rx_slab, pde_ry_slab, pde_rz_slab))
        pde_r_vac  = self.concat((pde_rx_vac, pde_ry_vac, pde_rz_vac))
        
        # 乘以mask, mask.shape=(bs,1), pde_r_slab.shape=(bs,3), pde_r_vac.shape=(bs,3)
        # 通过broadcast进行扩充乘法
        ones = ms_np.ones_like(mask)
        domain_r = mask * pde_r_slab + (ones - mask) * pde_r_vac
        
        loss_domain = self.reduce_mean(self.l2_loss(domain_r, self.zeros_like(domain_r)))
        
        # 无源散度为0
        Ex_x = self.grad(data, 0, 0, u) 
        Ey_y = self.grad(data, 1, 1, u)
        Ez_z = self.grad(data, 2, 2, u)
        no_source_r = Ex_x + Ey_y + Ez_z  # 无源，散度为0
        
        loss_no_source = self.reduce_mean(self.l2_loss(no_source_r, self.zeros_like(no_source_r)))
        
        return loss_domain + loss_no_source
    
    
    @ms_function
    def get_boundary_loss(self, u, data):
        """边界损失"""
        coord_min = self.config["coord_min"]
        coord_max = self.config["coord_max"]
        batch_size, _ = data.shape
        # 只选择部分数据
        mask = ms_np.zeros(shape=(batch_size, 14), dtype=ms.dtype.float32)
        
        mask[:, 0] = ms_np.where(ms_np.isclose(data[:, 0], coord_max[0]), 1.0, 0.0) # right
        mask[:, 1] = ms_np.where(ms_np.isclose(data[:, 0], coord_max[0]), 1.0, 0.0) # right

        mask[:, 2] = ms_np.where(ms_np.isclose(data[:, 1], coord_min[1]), 1.0, 0.0) # bottom     
        mask[:, 3] = ms_np.where(ms_np.isclose(data[:, 1], coord_min[1]), 1.0, 0.0) # bottom
        mask[:, 4] = ms_np.where(ms_np.isclose(data[:, 1], coord_min[1]), 1.0, 0.0) # bottom

        mask[:, 5] = ms_np.where(ms_np.isclose(data[:, 1], coord_max[1]), 1.0, 0.0) # top
        mask[:, 6] = ms_np.where(ms_np.isclose(data[:, 1], coord_max[1]), 1.0, 0.0) # top
        mask[:, 7] = ms_np.where(ms_np.isclose(data[:, 1], coord_max[1]), 1.0, 0.0) # top

        mask[:, 8] = ms_np.where(ms_np.isclose(data[:, 2], coord_min[2]), 1.0, 0.0) # back
        mask[:, 9] = ms_np.where(ms_np.isclose(data[:, 2], coord_min[2]), 1.0, 0.0) # back
        mask[:, 10] = ms_np.where(ms_np.isclose(data[:, 2], coord_min[2]), 1.0, 0.0) # back

        mask[:, 11] = ms_np.where(ms_np.isclose(data[:, 2], coord_max[2]), 1.0, 0.0) # front
        mask[:, 12] = ms_np.where(ms_np.isclose(data[:, 2], coord_max[2]), 1.0, 0.0) # front
        mask[:, 13] = ms_np.where(ms_np.isclose(data[:, 2], coord_max[2]), 1.0, 0.0) # front
        
        # 右面的ABC损失
        ## n x \nabla x E = 0 ==> dEz/dy - dEy/dz = 0
        # Ez_y = self.grad(data, 1, 2, u)
        # Ey_z = self.grad(data, 2, 1, u)
        
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

        bc_r_all = self.concat((bc_r_right, bc_r_round))
        bc_r = self.mul(bc_r_all, mask)
        
        loss_bc = self.reduce_mean(self.l2_loss(bc_r, self.zeros_like(bc_r)))
        return loss_bc

    
    @ms_function
    def get_port_loss(self, u, data):
        """波导管损失"""
        """waveguide-port loss"""
        uz = u[:, 2:3] # Ez
        label = data[:, 3:4] # 第0,1,2为输入的点，第3-6列为label
        # Ex=0, Ey=0, Ez=label
        waveguide = self.concat([u[:, 0:1], u[:, 1:2], uz - label])
        
        loss_port = self.reduce_mean(self.l2_loss(waveguide, self.zeros_like(waveguide)))
        return loss_port
    
    