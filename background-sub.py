'''
This code has been modified from https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
'''

import cv2 as cv
import argparse
import numpy as np
from create_video import create_video

parser1 = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser1.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='video_maskVideo/projectmoving.avi')
parser1.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args1 = parser1.parse_args()

parser2 = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser2.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='video_maskVideo/maskmoving.avi')
parser2.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args2 = parser2.parse_args()

# mask_border = cv.imread('border_mask.jpg');

if args1.algo == 'MOG2':
    backSub1 = cv.createBackgroundSubtractorMOG2()
    backSub2 = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture1 = cv.VideoCapture(cv.samples.findFileOrKeep(args1.input))
capture2 = cv.VideoCapture(cv.samples.findFileOrKeep(args2.input))
if not capture1.isOpened():
    print('Unable to open: ' + args1.input)
    exit(0)
mask_array = []

foreground_frames = []
background_frames = []
cnt = 1
prev_mask = None
while True:
    ret, frame = capture1.read()
    ret, mask = capture2.read()
    if frame is None:
        break
    # frame = cv.blur(frame, (10, 10))

    # element = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    # frame = cv.morphologyEx(frame, cv.MORPH_OPEN, element)

    fgMask = backSub1.apply(frame)
    # fgMask = fgMask * 0
    if prev_mask is not None:
        fgMask = fgMask * prev_mask
    fgMask = fgMask * (mask[:, :, 0] / 255)
    # fgMask = fgMask * (mask[:, :, 0] / 255) * (mask_border_resize / 255)

    # fgMask = fgMask * (mask[:, :, 0]/255)
    # fgMask = cv.blur(fgMask, (3, 3))

    element = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, element)
    moving_obj = np.zeros(frame.shape)
    moving_obj[:, :, 0] = fgMask/255 * frame[:, :, 0]/255
    moving_obj[:, :, 1] = fgMask/255 * frame[:, :, 1]/255
    moving_obj[:, :, 2] = fgMask/255 * frame[:, :, 2]/255

    background = np.zeros(frame.shape)
    background[:, :, 0] = (255 - fgMask)/255 * frame[:, :, 0]/255
    background[:, :, 1] = (255 - fgMask)/255 * frame[:, :, 1]/255
    background[:, :, 2] = (255 - fgMask)/255 * frame[:, :, 2]/255
    #
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture1.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    prev_mask = mask[:, :, 0] / 255
    foreground_frames.append(moving_obj)
    cv.imwrite(f'./foreground_frames/foreground_frame{cnt}.jpg', moving_obj * 255)
    background_frames.append(background)
    cv.imwrite(f'./background_frames/background_frame{cnt}.jpg', background * 255)
    cnt += 1
    # cv.imshow('Frame', frame)
    # cv.imshow('FG Mask', fgMask)
    # cv.imshow('mask', mask[:, :, 0])
    # cv.imshow('moving_obj', moving_obj)
    # cv.imshow('background', background)
    # height, width = fgMask.shape
    # size = (width, height)
    # fgMask_Video = np.array([[fgMask], [fgMask], [fgMask]])
    #
    # fgMask_Video = fgMask_Video.reshape([height,width, 3])
    #
    # mask_array.append(fgMask_Video)
    # keyboard = cv.waitKey(500)
    # if keyboard == 'q' or keyboard == 27:
    #     break
# out = cv.VideoWriter('mask3.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, size)
#
# for i in range(len(mask_array)):
#     out.write(mask_array[i])
# out.release()
# print(foreground_frames[0])
create_video('./foreground_frames/foreground_frame*.jpg', 'foreground_video.avi')
create_video('./background_frames/background_frame*.jpg', 'background_video.avi')
