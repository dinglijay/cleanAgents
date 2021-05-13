import kornia
import torch
import numpy as np

from model import DIP_Net

THR_SSIM = 0.60
SSIM_WD_SIZE = 5
THR_JUMP_ITER = 50
K = 15
N_INTERPOLATE = 11
STEP_SIZE = 0.3

def torch_diff(x):
    return x[1:] - x[:-1]

def rec_by_detect(inputs, output, victim):
    # 1. Cross-boundary points detection and filter based on 
    # ssim and n_iter threshold --> indices_stripped
    ssims = kornia.ssim(inputs[1].expand_as(inputs), inputs, 5).mean(dim=(1,2,3))
    diff = torch_diff(torch.argmax(output[2:], dim=1))
    indices = torch.nonzero(diff)[-K:]+1
    indices_stripped = indices[ssims[2:][indices]>THR_SSIM] + 2
    indices_stripped = indices_stripped[indices_stripped>THR_JUMP_ITER]
    n_jump = indices_stripped.shape[0]

    # 2. If no cross-boundary point, reconstruct image by averaging the last K images
    if indices_stripped.numel() == 0:
        rec_img = inputs[-K:].mean(dim=0, keepdim=True)
        return rec_img

    # 3. Check the class jump at every cross-boundary point
    cls_l = torch.argmax(output, dim=1)[indices_stripped-1]
    cls_r = torch.argmax(output, dim=1)[indices_stripped]
    assert torch.all(cls_l != cls_r)

    # 4. Liner Interpolation between every cross-boundary point pairs --> imgs_linspace
    imgs_l = inputs[indices_stripped-1]
    imgs_r = inputs[indices_stripped]
    imgs_linspace = [imgs_l + (imgs_r - imgs_l) * i for i in np.linspace(0, 1, N_INTERPOLATE)]
    imgs_linspace = torch.vstack(imgs_linspace)

    # 5. Liner search and get the boundary images --> bd_imgs
    # logits with shape (K*N_INTERPOLATE, 10) or (n_jump*N_INTERPOLATE, 10)
    logits = []
    for chunk in torch.chunk(imgs_linspace, 5):
        logits.append(victim(chunk).detach())
    logits = torch.vstack(logits).detach()
    logits = torch.nn.functional.softmax(logits, dim=1)
    rec_idx = []
    for k, chunk in enumerate(torch.chunk(logits, n_jump)):
        idx = torch.argmin(abs(chunk[:, cls_l[k]] - chunk[:, cls_r[k]]))
        rec_idx.append(idx+k*N_INTERPOLATE)
    bd_imgs = imgs_linspace[torch.hstack(rec_idx)]

    # 6. Push the boundary images back, and return their average
    direction = inputs[1] -bd_imgs
    rec_imgs = bd_imgs - STEP_SIZE*direction
    rec_imgs = rec_imgs.mean(dim=0, keepdim=True)

    return rec_imgs

class Defender(torch.nn.Module):
    
    def __init__(self, victim, **params):
        super(Defender, self).__init__()
        self.dip_tracker = DIP_Net(**params)
        self.victim = victim
        self.cnt = 0

    def forward(self, adv_images):
        self.cnt += 1
        print(f'# DIP Optimization: {self.cnt}')

        dip_images = self.dip_tracker.tracing(adv_images.detach())
        inputs = torch.vstack((adv_images, adv_images, dip_images))

        output = []
        for chunk in torch.chunk(inputs, 10):
            output.append(self.victim(chunk).detach())
        output = torch.vstack(output)
        output = torch.nn.functional.softmax(output, dim=1)

        rec_img = rec_by_detect(inputs, output, self.victim)
        
        return rec_img.detach()
