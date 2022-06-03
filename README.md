
## Summary
OpenCV and Python program to stitch two input images.

## Usage Instructions
- Use the command
```
python stitch_images.py <dir to images> <output result name>
```
- Output images will be written inside the folder 'results'

Algorithm steps:

1. Read in the image
2. Use SIFT to extract the feature points of each image and the descriptor corresponding to each feature point
By matching feature point descriptors, find matching feature point pairs in the two images (there may be false matches here)
3. Use the RANSAC algorithm to eliminate false matches
4. Solve the equation system and calculate the Homograph homography transformation matrix
5. Perform affine transformation on one of the images through the Homograph homography transformation matrix
6.Stitching pictures


- Use the command
- To get the frames from a video.
``` python
python video2frames.py <dir to video>
```

- Get the commandline to run
``` python
python argCreator.py
```
- Cmd + V on the terminal and run the stitching_detailed.py to get stiched frames

- Get the video.The type you can choose video,mask,foreground,background
``` python
python create_video.py --input_type <type>
```
- To predict the midframe of the video and calculate the RMSE of difference with the Homography calculated from SIFT and Superglue.
- You can put your video in Rmse_analysis/video
``` python
cd Rmse_analysis
python mid_predict_H_Rmse.py <name of video>
```
- To get the one stiched images from images and draw the matching pairs
``` python
python stitch_images.py <dir to images> <output result name>
```
- To get the stiched images and the blended stiched image
``` python
python laplacian_blending.py <dir to images> <output result name>
```
- To run the adaptive GMM algorithm to get the foreground and background 
``` python
python background-sub.py
```
- Do matching with superglue and draw the matching pairs
``` python
cd superglue_matching
python match_pairs.py
```

- To run the Vibe algorithm to get the foreground and background
``` python
cd vibe
python vibe_demo.py
```
