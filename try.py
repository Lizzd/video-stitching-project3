import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class GaussianBlur(nn.Module):
    # 利用一个group convolution层实现高斯模糊模块
    def __init__(self, ksize, sigma=0, channels=3):
        super().__init__()
        padding = (ksize - 1) // 2
        self.conv = nn.Conv2d(channels, channels, ksize, 1, padding, groups=channels, bias=False,
                              padding_mode='reflect')
        self.init_weight(ksize, sigma, channels)

    @staticmethod
    @torch.no_grad()
    def getGaussianKernel(ksize, sigma=0):
        # 根据 kernel size 和 sigma 得到卷积核的权重
        if sigma <= 0:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        center = ksize // 2
        xs = (np.arange(ksize, dtype=np.float32) - center)
        kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))  # exp in numpy is faster than in torch or in math
        kernel = kernel1d[..., None] @ kernel1d[None, ...]
        kernel = torch.from_numpy(kernel)
        kernel = kernel / kernel.sum()
        return kernel.type(torch.float32)

    def init_weight(self, ksize, sigma, channels):
        # 初始化卷积核权重
        kernel = self.getGaussianKernel(ksize, sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.conv.weight.data = kernel

    def forward(self, img):
        return self.conv(img)


class LaplacianBlending(nn.Module):
    def __init__(self, ksize=3, sigma=0, channels=3, iters=4):
        super().__init__()
        self.gaussian = GaussianBlur(ksize, sigma, channels)
        self.iters = iters

    @staticmethod
    def scale(x, s):
        x = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=True)
        return x

    def down(self, x, y, mask):
        input = torch.cat((x, y))
        input_blur = self.gaussian(input)
        # input_blur = input # 如果想试验不做高斯模糊，则可以取消注释本行，并注释上一行
        input_blur_half = self.scale(input_blur, 0.5)
        # input_lap = input - self.scale(input_blur_half, 2)
        input_lap = input - input_blur_half.resize(input.shape)
        mask_half = self.scale(mask, 0.5)
        x_blur_half, y_blur_half = torch.chunk(input_blur_half, 2)
        x_lap, y_lap = torch.chunk(input_lap, 2)
        return x_blur_half, y_blur_half, x_lap, y_lap, mask_half

    @staticmethod
    def blend(x, y, mask):
        return x * mask + y * (1 - mask)

    def up(self, xy_blend, x_lap, y_lap, mask):
        out = self.scale(xy_blend, 2)
        diff = self.blend(x_lap, y_lap, mask)
        out = out + diff
        return out

    def forward(self, x, y, mask):
        x_laps = []
        y_laps = []
        masks = [mask]
        for it in range(self.iters):
            x, y, x_lap, y_lap, mask = self.down(x, y, mask)
            x_laps.append(x_lap)
            y_laps.append(y_lap)
            masks.append(mask)

        xy_blend = self.blend(x, y, masks[-1])
        for it in range(self.iters):
            idx = self.iters - 1 - it
            x_lap = x_laps[idx]
            y_lap = y_laps[idx]
            msk = masks[idx]
            xy_blend = self.up(xy_blend, x_lap, y_lap, msk)

        xy_blend = torch.clamp(xy_blend, 0.0, 1.0)
        return xy_blend


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    from torchvision import transforms as TF

    transform = TF.Compose([TF.ToTensor()])

    I = cv2.imread('./images/apple_orange/apple.png')
    J = cv2.imread('./images/apple_orange/orange.png')
    J = cv2.resize(J, [I.shape[1], I.shape[0]])
    mask = np.ones(J.shape)
    mask[:,int(I.shape[1]/2)] = 0

    I = transform(I).unsqueeze(0).contiguous()
    J = transform(J).unsqueeze(0).contiguous()
    mask = transform(mask).unsqueeze(0).contiguous()

    LB = LaplacianBlending()
    out = LB(I, J, mask)
    # out = torch.clamp(out, 0, 1)
    out = out.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    out = (out * 255).astype('uint8')[:, :, ::-1]
    plt.imshow(out)
    plt.show()
    print('END')
