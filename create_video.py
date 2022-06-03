import cv2
import numpy as np
import glob
import re
import argparse

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_video(pattern, output_name):
    img_array = []
    filenames = []
    for filename in glob.glob(pattern):
        filenames.append(filename)
    filenames.sort(key=natural_keys)
    for filename in filenames:
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

# create_video('./frames/fixed_frame*.jpg', 'projectmoving.avi')
# create_video('./frames/mask_fixed_frame*.jpg', 'maskmoving.avi')
# create_video('./foreground_frames/foreground_frame*.jpg', 'foreground_video.avi')
# create_video('./background_frames/background_frame*.jpg', 'background_video.avi')


TYPES = {
    'video': ['./frames/fixed_frame*.jpg', './video_maskVideo/projectmoving.avi'],
    'mask': ['./frames/mask_fixed_frame*.jpg', './video_maskVideo/maskmoving.avi'],
    'foreground': ['./foreground_frames/foreground_frame*.jpg', 'foreground_video.avi'],
    'background': ['./background_frames/background_frame*.jpg', 'background_video.avi']
}

parser = argparse.ArgumentParser(
    prog="stitching_detailed.py", description="Rgotation model images stitcher"
)
parser.add_argument(
    '--input_type',
    action='store',
    default='video',
    help="[video, mask, foreground, background]",
    type=str, dest='input_type'
)


if __name__ == '__main__':
    args = parser.parse_args()
    input_type = args.input_type
    print(input_type)
    pattern, output_name = TYPES[input_type]
    create_video(pattern, output_name)