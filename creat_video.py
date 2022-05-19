import cv2
import numpy as np
import glob

img_array = []
# for filename in glob.glob('./frames/fixed_frame*.jpg'):
for i in range(43):
    # filename = './frames/fixed_frame' + str(i) + '.jpg'
    filename = './mask/mask_fixed_frame' + str(i) + '.jpg'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('maskmoving.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
# out = cv2.VideoWriter('projectmoving.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()