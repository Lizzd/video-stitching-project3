import cv2
import matplotlib.pyplot as plt


def laplcian_blending(I, J, mask):
    current_I = I
    current_J = J

    diffIs = []
    diffJs = []
    masks = [mask]

    for i in range(4):
        h, w, c = current_I.shape
    I_half = cv2.resize(current_I, (w // 2, h // 2))
    I_up = cv2.resize(I_half, (w, h))
    diff_I = current_I - I_up
    current_I = I_half

    J_half = cv2.resize(current_J, (w // 2, h // 2))
    J_up = cv2.resize(J_half, (w, h))
    diff_J = current_J - J_up
    current_J = J_half

    mask = cv2.resize(mask, (w // 2, h // 2))

    diffIs.append(diff_I)
    diffJs.append(diff_J)
    masks.append(mask)

    current_merge = current_I * masks[4] + current_J * (1 - masks[4])

    rgb_merges = [current_merge.astype('uint8')[:, :, ::-1]]
    diff_merges = []

    for i in range(4):
        h, w, c = current_merge.shape
    current_merge_up = cv2.resize(current_merge, (w * 2, h * 2))
    diff_up = diffIs[4 - i - 1] * masks[4 - i - 1] + diffJs[4 - i - 1] * (1 - masks[4 - i - 1])
    diff_up_show = ((diff_up - diff_up.min()) / (diff_up.max() - diff_up.min()) * 255).astype('uint8')
    diff_merges.append(diff_up_show)
    current_merge = current_merge_up + diff_up
    rgb_merges.append(current_merge.astype('uint8')[:, :, ::-1])

    fig, axs = plt.subplots(4, 2, constrained_layout=True)
    for i in range(4):
        axs[i, 0].imshow(rgb_merges[i])
    axs[i, 0].set_title('({}) 1/{}'.format(i * 2 + 1, 2 ** (4 - i)), y=-0.27)
    axs[i, 0].axis('off')

    axs[i, 1].imshow(diff_merges[i])
    axs[i, 1].set_title('({})'.format(i * 2 + 2), y=-0.27)
    axs[i, 1].axis('off')

    # plt.figure()
    # plt.imshow(rgb_merges[4])
    # plt.axis('off')
    # plt.show()
    return rgb_merges[4]
# I = cv2.imread('apple.png').astype('float32')
# J = cv2.imread('orange.png').astype('float32')
# mask = cv2.imread('mask.png').astype('float32') / 255.0
