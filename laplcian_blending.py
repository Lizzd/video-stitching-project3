import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# def laplcian_blending(left_image, right_image):
#     # imgApple = cv2.imread("./images/apple.png")[:, :, ::-1]
#     # imgOrange = cv2.imread("./images/orange.png")[:, :, ::-1]
#     imgApple = left_image
#     imgOrange = cv2.flip(right_image, 1)
#
#     # imgOrange = imgOrange[3:-3, 1:-2]
#     imgOrange = cv2.resize(imgOrange, [imgApple.shape[1], imgApple.shape[0]])
#     mask_size = imgOrange.shape[1]
#
#     cv2.imshow('left_image', imgApple)
#     cv2.imshow('right_image', imgOrange)
#
#     # plt.figure(figsize=(10, 5))
#     # plt.subplot(1, 2, 1)
#     # plt.title("imgApple")
#     # plt.imshow(imgApple, cmap=cm.gray)
#     # plt.subplot(1, 2, 2)
#     # plt.title("imgOrange")
#     # plt.imshow(imgOrange, cmap=cm.gray)
#     # plt.show()
#
#     imgAppleMask = np.ones([imgApple.shape[0], imgApple.shape[1] * 2], np.uint16) * 255
#     imgOrangeMask = np.ones([imgOrange.shape[0], imgOrange.shape[1] * 2], np.uint16) * 255
#     imgAppleMask[:, int(np.floor(mask_size)):-1] = 0
#     imgOrangeMask[:, 0:int(np.floor(mask_size))] = 0
#
#     # plt.figure(figsize=(10, 5))
#     # plt.subplot(1, 2, 1)
#     # plt.title("imgAppleMask")
#     # plt.imshow(imgAppleMask, cmap=cm.gray)
#     # plt.subplot(1, 2, 2)
#     # plt.title("imgOrangeMask")
#     # plt.imshow(imgOrangeMask, cmap=cm.gray)
#     # plt.show()
#
#     blender = cv2.detail_MultiBandBlender()
#     roiRect = (0, 0, imgApple.shape[1] * 2, imgApple.shape[0])
#     blender.prepare(roiRect)
#     blender.feed(imgApple, imgAppleMask, (0, 0))
#     blender.feed(imgOrange, imgOrangeMask, (0, 0))
#     imgBlendAO = imgApple.copy()
#     dst, dst_mask = blender.blend(imgBlendAO, imgAppleMask)
#     # plt.imshow(dst)
#     plt.imshow(dst.astype(np.uint8))
#     plt.show()


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
        input_lap = input - self.scale(input_blur_half, 2)
        # input_lap = input - cv2.resize(input_blur_half, [input.shape[1], input.shape[0]])
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


import cv2
import matplotlib.pyplot as plt
from torchvision import transforms as TF


def laplcian_blending(image, mask):
    transform = TF.Compose([TF.ToTensor()])

    # I = cv2.imread('apple.png')
    # J = cv2.imread('orange.png')
    # mask = cv2.imread('mask.png')
    I = image
    J = image
    mask = mask
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

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
#
#
# # I = cv2.imread('apple.png').astype('float32')
# # J = cv2.imread('orange.png').astype('float32')
# # mask = cv2.imread('mask.png').astype('float32') / 255.0
# # def laplcian_blending(image1, image2):
# def laplcian_blending(image, mask):
#     I = image
#     J = image
#
#     current_I = I  # left
#     current_J = J  # right
#     diffIs = []
#     diffJs = []
#     # cv2.imshow('I',current_I)
#     # current_J = cv2.resize(J, [I.shape[1], I.shape[0]])
#     # cv2.imshow('J',current_J)
#     # diffIs = []
#     # diffJs = []
#     # mask = np.ones(I.shape)
#     # mask[:, int(I.shape[1] / 2), :] = 0
#     # mask = mask.astype('float32') / 255.0
#     masks = [mask]
#     for i in range(4):
#         h, w, c = current_I.shape
#         I_half = cv2.resize(current_I, (w // 2, h // 2))
#         I_up = cv2.resize(I_half, (w, h))
#         diff_I = current_I - I_up
#         current_I = I_half
#
#         J_half = cv2.resize(current_J, (w // 2, h // 2))
#         J_up = cv2.resize(J_half, (w, h))
#         diff_J = current_J - J_up
#         current_J = J_half
#
#         mask = cv2.resize(mask, (w // 2, h // 2))
#
#         diffIs.append(diff_I)
#         diffJs.append(diff_J)
#         masks.append(mask)
#
#     current_merge = current_I * masks[4] + current_J * (1 - masks[4])
#
#     rgb_merges = [current_merge.astype('uint8')[:, :, ::-1]]
#     diff_merges = []
#
#     for i in range(4):
#         h, w, c = current_merge.shape
#         current_merge_up = cv2.resize(current_merge, (w * 2, h * 2))
#         diff_up = diffIs[4 - i - 1] * masks[4 - i - 1] + diffJs[4 - i - 1] * (1 - masks[4 - i - 1])
#         diff_up_show = ((diff_up - diff_up.min()) / (diff_up.max() - diff_up.min()) * 255).astype('uint8')
#         diff_merges.append(diff_up_show)
#         diff_up = cv2.resize(diff_up, [current_merge_up.shape[0], current_merge_up.shape[1]])
#         current_merge = current_merge_up + diff_up
#         rgb_merges.append(current_merge.astype('uint8')[:, :, ::-1])
#
#     fig, axs = plt.subplots(4, 2, constrained_layout=True)
#     for i in range(4):
#         axs[i, 0].imshow(rgb_merges[i])
#         axs[i, 0].set_title('({}) 1/{}'.format(i * 2 + 1, 2 ** (4 - i)), y=-0.27)
#         axs[i, 0].axis('off')
#
#         axs[i, 1].imshow(diff_merges[i])
#         axs[i, 1].set_title('({})'.format(i * 2 + 2), y=-0.27)
#         axs[i, 1].axis('off')
#
#     plt.figure()
#     plt.imshow(rgb_merges[4])
#     plt.axis('off')
#     plt.show()

#     # imgApple = cv2.imread("./images/apple.png")[:, :, ::-1]
#     # imgOrange = cv2.imread("./images/orange.png")[:, :, ::-1]
# image1 = cv2.imread("./images/apple_orange/apple.png")
# image2 = cv2.imread("./images/apple_orange/orange.png")
# laplcian_blending(image1, image2)
