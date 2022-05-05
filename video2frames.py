import os
import sys
from turtle import width

import cv2
import numpy as np


def save_image(img, name):
    result_image_name = os.path.join('results/', f'result_{name}.jpg')
    cv2.imwrite(result_image_name, img)


def show_image(img, title, width=1024, inter=cv2.INTER_AREA):
    (h, w) = img.shape[:2]
    r = width / float(w)
    dim = (width, int(h * r))
    resized = cv2.resize(img, dim, interpolation=inter)
    cv2.imshow(title, resized)

def getFrame(path):
    vidcap = cv2.VideoCapture(path)     #test1.mp4
    success,image = vidcap.read()
    count = 0
    i = 0
    frames = []
    while success:
        if (count % 10 == 0):
            # image = cv2.resize(image, (640, 360))
            cv2.imwrite("./frames/frame%d.jpg" % (i), image)     # save frame as JPEG file, could delete later  
            # cv2.imwrite("./frames/frame%d.jpg" % (count), image)     # save frame as JPEG file, could delete later  
            frames.append(image)
            i += 1
        success,image = vidcap.read()
        count += 1
    return frames

# Main function definition
def main():
    assert len(sys.argv) == 2, 'Error: Please provide the path to the video file'

    getFrame(sys.argv[1])


# Call main function
if __name__ == '__main__':
    main()
