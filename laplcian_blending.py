'''
This code has been modified from https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html
'''

import os
import sys
from turtle import width

import cv2
import numpy as np

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

    mask = np.ones(result_img.shape, dtype=result_img.dtype)
    result_img[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[
        0]] = img1  # image2 is transformed to result_image through perspective transformation,
    # image1 occupies the right side of result_image, and the overlapping part is covered by image1
    mask[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[
        0]] = 0
    warpped_img2 = cv2.warpPerspective(img2, transform_array.dot(M),
                                       (x_max - x_min, y_max - y_min))
    img_A = np.zeros(result_img.shape, dtype=result_img.dtype)
    img_B = np.zeros(result_img.shape, dtype=result_img.dtype)
    img_A += warpped_img2
    img_B += (1 - mask) * result_img

    assert img_A.shape == img_B.shape
    h, w = img_A.shape[:2]
    w2 = 2 ** np.ceil(np.log2(w))
    h2 = 2 ** np.ceil(np.log2(h))
    w2 = int(w2)
    h2 = int(h2)
    print('w, h', w, h)
    print('w2, h2', w2, h2)
    print('w2 - w, h2 - h', w2 - w, h2 - h)

    img_A = cv2.copyMakeBorder(src=img_A, top=0, bottom=h2 - h, left=0, right=w2 - w, borderType=cv2.BORDER_CONSTANT,
                               value=0)
    img_B = cv2.copyMakeBorder(src=img_B, top=0, bottom=h2 - h, left=0, right=w2 - w, borderType=cv2.BORDER_CONSTANT,
                               value=0)
    mask = cv2.copyMakeBorder(src=mask, top=0, bottom=h2 - h, left=0, right=w2 - w, borderType=cv2.BORDER_CONSTANT,
                              value=0)

    print(img_A.shape)
    print(img_B.shape)
    cv2.imshow('img_A', img_A)
    cv2.imshow('img_B', img_B)
    # cv2.waitKey()

    # laplcian_blending
    # generate Gaussian pyramid for A
    G = img_A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = img_B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in range(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    # image with direct connecting each half
    real = np.hstack((img_A[:, :cols // 2], img_B[:, cols // 2:]))
    # Return the result
    return result_img, ls_

    # return result_img


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


# Main function definition
def main():
    assert len(sys.argv) == 3, 'Error: Please provide the path to the directory of the images AND the output name'
    assert os.path.isdir(sys.argv[1]), 'Error: Please provide the valid path to the directory of the images'

    first_img = True
    input_images = []
    for filename in os.listdir(sys.argv[1]):
        f = os.path.join(sys.argv[1], filename)
        if os.path.isfile(f):
            if first_img:
                img1 = cv2.imread(f)
                input_images.append(img1)
                first_img = False
                continue

            img2 = cv2.imread(f)
            input_images.append(img2)

            # Show matched keypoint of two images
            # draw_matched_keypoint(img1, img2)

            imgs, [img1_key, img2_key] = stitch_images(img1, img2)
            img1, img3 = imgs
            # img1, [img1_key, img2_key] = stitch_images(img1, img2)
    save_image(img1, sys.argv[2])
    # save_image(img2, sys.argv[2])
    # Show the resulting image
    save_image(img3, f'lap_{sys.argv[2]}')
    show_image(img3, 'lap Result')
    show_image(img1, 'Result')
    cv2.waitKey()


# Call main function
if __name__ == '__main__':
    main()
