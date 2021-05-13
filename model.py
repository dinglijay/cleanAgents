import kornia
import torch
import numpy as np

from dip_trace import get_dip_net
from dip_trace.dip_utils import get_noise, get_params

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

class DIP_Net():

    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'
    LR = 0.01
    OPTIMIZER='adam' # 'LBFGS'

    def __init__(self, num_iter=1000, 
                       num_save=200,
                       reg_noise_std=1/30.0,
                       input_depth=4,
                       n33d_n33u_n11s_ns=[32, 32, 4, 3],
                       x_noise_std=0.0,
                       loss='mse'):
        self.num_iter = num_iter
        self.save_iters = np.linspace(0, num_iter, num_save+1).astype(np.int).tolist()
        self.reg_noise_std = reg_noise_std
        self.input_depth = input_depth
        self.x_noise_std = x_noise_std
        self.build_net(n33d_n33u_n11s_ns, loss)

    def build_net(self, net_params, loss):
        self.params = {'skip_n33d': net_params[0], 
                       'skip_n33u': net_params[1], 
                       'skip_n11':  net_params[2],
                       'num_scales':net_params[3]}
        self.net = get_dip_net(self.input_depth,
                               DIP_Net.pad,
                               upsample_mode='bilinear',
                               **self.params).type(dtype)
        if loss=='mse': self.loss = torch.nn.MSELoss().type(dtype)
        if loss=='ssim': self.loss = kornia.losses.SSIMLoss(window_size=5).type(dtype)
        s  = sum([np.prod(list(p.size())) for p in self.net.parameters()]); 
        print(f'Num_iter/save: {self.num_iter}/{len(self.save_iters)}')
        print(f'Net: {net_params}')
        print(f'Input_depth: {self.input_depth}')
        print(f'Reg/x_noise_std: {self.reg_noise_std}/{self.x_noise_std}')
        print(f'Net Loss: {loss}')
        print(f'Number of params: {s}')

    def tracing(self, img_noisy_torch=None, reinit_net=True):
        if reinit_net:
            self.net = get_dip_net(self.input_depth,
                               DIP_Net.pad,
                               upsample_mode='bilinear',
                               **self.params).type(dtype)
        net = self.net

        net_input = get_noise(self.input_depth,
                              DIP_Net.INPUT, 
                              img_noisy_torch.shape[-2:]).type(dtype).detach()
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        img_saved = img_noisy_torch.detach().clone()
        noise_img = img_noisy_torch.detach().clone()

        params = get_params(DIP_Net.OPT_OVER, net, net_input)
        optimizer = torch.optim.Adam(params, lr=DIP_Net.LR)
        save_imgs = []
        for j in range(self.num_iter):
            if self.reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * self.reg_noise_std)
            if self.x_noise_std > 0:
                img_noisy_torch = img_saved + (noise_img.normal_() * self.x_noise_std)
            out = net(net_input)
            
            total_loss = self.loss(out, img_noisy_torch)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if j+1 in self.save_iters:
                out = net(net_input.detach())
                save_imgs.append(out.detach())
        return torch.vstack(save_imgs)

