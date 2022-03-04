
## Summary
OpenCV and Python program to stitch two input images.
some code from https://github.com/pavanpn/Image-Stitching
## Usage Instructions
- Use the command
```
python stitch_images.py <FirstImage> <SecondImage>
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

