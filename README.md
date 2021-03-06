
## Summary
OpenCV and Python program to stitch two input images.

Stitching Video demo1
![image](https://github.com/Lizzd/video-stitching-project3/blob/812a840c389116ec1fddc52910dcc7acd199c962/GIF/projectmoving_people.gif)
![foreground](https://github.com/Lizzd/video-stitching-project3/blob/1f6f3d75dcc63bb805da45d9963c3a67ce3c5826/GIF/foreground_video_people.gif)
![background](https://github.com/Lizzd/video-stitching-project3/blob/1f6f3d75dcc63bb805da45d9963c3a67ce3c5826/GIF/background_video_people.gif)

Stitching Video demo2
![image](https://github.com/Lizzd/video-stitching-project3/blob/812a840c389116ec1fddc52910dcc7acd199c962/GIF/projectmoving_rlc.gif)
![image](https://github.com/Lizzd/video-stitching-project3/blob/812a840c389116ec1fddc52910dcc7acd199c962/GIF/foreground_video_rlc.gif)
![background](https://github.com/Lizzd/video-stitching-project3/blob/1f6f3d75dcc63bb805da45d9963c3a67ce3c5826/GIF/background_video_rlc.gif)

Stitching Video demo3
![image](https://github.com/Lizzd/video-stitching-project3/blob/972aa19c42229e864cdeeb0dbb7d42189e96b14c/GIF/projectmoving_square.gif)
![image](https://github.com/Lizzd/video-stitching-project3/blob/972aa19c42229e864cdeeb0dbb7d42189e96b14c/GIF/foreground_video_square.gif)
![image](https://github.com/Lizzd/video-stitching-project3/blob/972aa19c42229e864cdeeb0dbb7d42189e96b14c/GIF/background_video_square.gif)
## Usage Instructions
All the videos used in the report can be found in [this](https://github.com/Lizzd/video-stitching-project3/tree/main/Rmse_analysis/video).

Algorithm steps:

1. Read in the image
2. Use SIFT to extract the feature points of each image and the descriptor corresponding to each feature point
By matching feature point descriptors, find matching feature point pairs in the two images (there may be false matches here)
3. Use the RANSAC algorithm to eliminate false matches
4. Solve the equation system and calculate the Homograph homography transformation matrix
5. Perform affine transformation on one of the images through the Homograph homography transformation matrix
6.Stitching pictures


- Use the command
- REMOVE all the images from the ./frames first!
- To get the frames from a video.
- After running this script, the frames of the video will be saved in the ./frames
``` python
python video2frames.py <dir to video>
```

- To run video stitching script
- First, run `argCreator.py` to get the commandline for running the `stitching_detailed.py`
- Cmd + V on the terminal and run the stitching_detailed.py to get stiched frames
``` python
python argCreator.py

# Cmd + V / Crtl + V and Enter
```

- To run the adaptive GMM algorithm to get the foreground and background 
- REMOVE all the images in the ./background_frames AND ./foreground_frames
- RUN `stitching_detailed.py` first
- After running this script, the `background_video.avi` and `foreground_video.avi` should be available at the root directory.
``` python
python create_video.py --input_type video # the stitced video will be saved at './video_maskVideo/projectmoving.avi'
python create_video.py --input_type mask  # the mask video where each frame is being places will be saved at './video_maskVideo/maskmoving.avi'
python background-sub.py
```

- Get the video.The type you can choose video, mask, foreground, background
- To make a video from the `stitching_detailed.py` use `video`
``` python
python create_video.py --input_type <type>

# Creating a result stitching video 
python create_video.py --input_type video
# Creating a mask video, this must be done every time before running background-sub.py
python create_video.py --input_type mask

```

- To predict the midframe of the video and calculate the RMSE of difference with the Homography calculated from SIFT and Superglue.
- You must put your video in Rmse_analysis/video
- `<name of the video>` should only be the name without the file extension.
``` python
cd Rmse_analysis
python mid_predict_H_Rmse.py <name of video>
```
- To get the one stiched images from images and draw the matching pairs
- This is not the script for video stitching. This is naive algorithm described in the report.
``` python
python stitch_images.py <dir to images> <output result name>
```
- To get the stiched images and the blended stiched image
``` python
python laplacian_blending.py <dir to images> <output result name>
```

- Do matching with superglue and draw the matching pairs
``` python
cd superglue_matching
python match_pairs.py
```

- To run the Vibe algorithm to get the foreground and background with demo video
``` python
cd vibe
python vibe_demo.py
```
