import os
import sys
from turtle import width

import cv2
import numpy as np
import laplcian_blending
from ransac import ransac
from Homography import Homography


# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):
    # Get width and height of input images
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp,
                                         M)  # Image 2 is converted to the same coordinate system as image 1 using Homography mat

    # Resulting dimensions

    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)  # Scale out by one pixel
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))
    cv2.imshow('result1',result_img)
    result_img[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[
        0]] = img1  # image2 is transformed to result_image through perspective transformation,
    # image1 occupies the right side of result_image, and the overlapping part is covered by image1
    cv2.imshow('result2', result_img)
    ##laplcian_blending
    # width = result_img.shape()[0]
    # height = result_img.shape()[1]
    # mask = np.ones(width, height)
    # mask[transform_dist[1]:w1 + transform_dist[1],
    # transform_dist[0]:h1 + transform_dist[
    #     0]] = 0
    # left_imag = result_img[:transform_dist[1], :transform_dist[0]]
    # right_imag = result_img[transform_dist[1]:w1 + transform_dist[1], transform_dist[0]:h1 + transform_dist[
    #     0]]
    # lap_result_img  = laplcian_blending(left_imag, right_imag, mask)

    # Return the result
    # return result_img, lap_result_img

    return result_img


# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):
    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Extract keypoints and descriptors
    kp1, d1 = sift.detectAndCompute(img1, None)
    kp2, d2 = sift.detectAndCompute(img2, None)

    # Bruteforce matcher on the descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # k1 and k2 match each other to find the smallest distance pair
    knnMatches = bf.knnMatch(d1, d2, k=2)  # one keypoint match two other keypoints

    # Make sure that the matches are good
    good_ratio = 0.8  # Source: stackoverflow
    good_matches = []
    for m1, m2 in knnMatches:
        # Add to array only if it's a good match
        if m1.distance < good_ratio * m2.distance:
            good_matches.append(m1)

    # Mimnum number of matches
    min_matches = 8
    if len(good_matches) > min_matches:

        # Array to store matching points
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for match in good_matches:
            img1_pts.append(kp1[match.queryIdx].pt + (1,))
            img2_pts.append(kp2[match.trainIdx].pt + (1,))
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 3)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 3)

        # Compute homography matrix
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)  # x2 = M*x1  many times choose
        # inlier = ransac(img1_pts, img2_pts, 5.0, 1000)  # x2 = M*x1  many times choose
        # M = Homography(inlier)
        # four points to calculate and use RANSAC to choose the best one
        return M, img1, img2
    else:
        print('Error: Not enough matches')
        exit()


# Equalize Histogram of Color Images improve image contrast
def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # Avoid losing information in integer calculations
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img


match_count = 0


def draw_matched_keypoint(img1, img2, wait=False):
    global match_count
    img1 = img1.copy()
    img2 = img2.copy()

    sift = cv2.SIFT_create(100)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # draw the keypoints
    img1 = cv2.drawKeypoints(image=img1, keypoints=kp1, outImage=img1, color=(255, 0, 255),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2 = cv2.drawKeypoints(image=img2, keypoints=kp2, outImage=img2, color=(255, 0, 255),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.waitKey(20)

    bf = cv2.BFMatcher()
    knnMatches = bf.knnMatch(des1, des2, k=2)
    print(type(knnMatches), len(knnMatches), knnMatches[0])
    # # Get the first descriptor in img1 with the most matching descriptor in img2 with the smallest distance
    dMatch0 = knnMatches[0][0]
    # # Get the first descriptor in img1 that matches the next descriptor in img2, followed by the distance
    dMatch1 = knnMatches[0][1]
    print('knnMatches', dMatch0.distance, dMatch0.queryIdx, dMatch0.trainIdx)
    print('knnMatches', dMatch1.distance, dMatch1.queryIdx, dMatch1.trainIdx)
    # Make sure that the matches are good
    goodMatches = []
    minRatio = 0.8
    for m, n in knnMatches:
        if m.distance / n.distance < minRatio:
            goodMatches.append([m])

    print(len(goodMatches))
    sorted(goodMatches, key=lambda x: x[0].distance)
    # # draw verified matches
    outImg = None
    outImg = cv2.drawMatchesKnn(img1, kp1, img2, kp2, goodMatches, outImg, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    match_count += 1
    show_image(outImg, f'Match {match_count}')
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def stitch_images(img1, img2):
    # Equalize histogram
    img1 = equalize_histogram_color(img1)
    img2 = equalize_histogram_color(img2)

    # Use SIFT to find keypoints and return homography matrix
    M, img1_key, img2_key = get_sift_homography(img1, img2)

    # Stitch the images together using homography matrix
    return get_stitched_image(img2, img1, M), [img1_key, img2_key]


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
    frames = []
    while success:
        if (count % 15 == 0):
            cv2.imwrite("./frames/frame%d.jpg" % (count / 15), image)     # save frame as JPEG file, could delete later  
            frames.append(image)
        success,image = vidcap.read()
        count += 1
    return frames

# Main function definition
def main():
    assert len(sys.argv) == 3, 'Error: Please provide the path to the directory of the images AND the output name'
    # assert os.path.isdir(sys.argv[1]), 'Error: Please provide the valid path to the directory of the images'

    first_img = True
    input_images = getFrame(sys.argv[1])
    for i in range(len(input_images)):
        if first_img:
            img1 = input_images[i]
            first_img = False
            continue

        img2 = input_images[i]

        # Show matched keypoint of two images
        draw_matched_keypoint(img1, img2)

        # img1, img3, [img1_key, img2_key] = stitch_images(img1, img2)
        img1, [img1_key, img2_key] = stitch_images(img1, img2)
            
    save_image(img1, sys.argv[2])
    # save_image(img2, sys.argv[2])
    # Show the resulting image
    # show_image((img3, 'lap Result'))
    show_image(img1, 'Result')
    cv2.waitKey()


# Call main function
if __name__ == '__main__':
    main()
