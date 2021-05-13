import torch
import torchvision

class Normalize(torch.nn.Module) :
    def __init__(self, mean, std, scale=False) :
        super(Normalize, self).__init__()
        self.scale = scale
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        if self.scale:
            input = input/255.0
        return (input - mean) / std


# victim model
def get_victim(dataset='imagenet', net='resnet101', device='cuda'):

    print(f'Loading {net} network pretrained on {dataset} dataset')

    from torchvision.models import inception_v3, resnet50, resnet101, resnet152

    mean =  [0.485, 0.456, 0.406]
    std =   [0.229, 0.224, 0.225]

    victim = torch.nn.Sequential(
        Normalize(mean, std), 
        eval(net)(pretrained=True)
    ).to(device).eval()

    return victim

