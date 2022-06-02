import cv2
import numpy as np
import glob
import re

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


img_array = []
filenames = []
for filename in glob.glob('./frames/fixed_frame*.jpg'):
    filenames.append(filename)
filenames.sort(key=natural_keys)
for filename in filenames:
# for i in range(43):
    print(filename)
    # filename = './frames/fixed_frame' + str(i) + '.jpg'
    # filename = './mask/mask_fixed_frame' + str(i) + '.jpg'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

# out = cv2.VideoWriter('maskmoving.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
out = cv2.VideoWriter('projectmoving.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()