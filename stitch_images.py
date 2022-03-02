import sys
import cv2
import numpy as np


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
    result_img[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[
        0]] = img1  # image2 is transformed to result_image through perspective transformation,
    # image1 occupies the right side of result_image, and the overlapping part is covered by image1

    # Return the result
    return result_img


# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):
    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Extract keypoints and descriptors
    kp1, d1 = sift.detectAndCompute(img1, None)
    kp2, d2 = sift.detectAndCompute(img2, None)
    # # draw the keypoints
    # img1 = cv2.drawKeypoints(image=img1, keypoints=kp1, outImage=img1, color=(255, 0, 255),
    #                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img2 = cv2.drawKeypoints(image=img2, keypoints=kp2, outImage=img2, color=(255, 0, 255),
    #                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.waitKey(20)
    # Bruteforce matcher on the descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # k1 and k2 match each other to find the smallest distance pair
    knnMatches = bf.knnMatch(d1, d2, k=2)  # one keypoint match two other keypoints

    # # Get the first descriptor in img1 with the most matching descriptor in img2 with the smallest distance
    # dMatch0 = knnMatches[0][0]
    # # Get the first descriptor in img1 that matches the next descriptor in img2, followed by the distance
    # dMatch1 = knnMatches[0][1]
    # print('knnMatches', dMatch0.distance, dMatch0.queryIdx, dMatch0.trainIdx)
    # print('knnMatches', dMatch1.distance, dMatch1.queryIdx, dMatch1.trainIdx)
    # Make sure that the matches are good
    good_ratio = 0.8  # Source: stackoverflow
    good_matches = []
    for m1, m2 in knnMatches:
        # Add to array only if it's a good match
        if m1.distance < good_ratio * m2.distance:
            good_matches.append(m1)
    # draw verified matches
    # print(len(good_matches))
    # sorted(good_matches, key=lambda x: x[0].distance)
    # outImg = None
    # outImg = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, outImg, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # cv2.imshow('matche', outImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Mimnum number of matches
    min_matches = 8
    if len(good_matches) > min_matches:

        # Array to store matching points
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for match in good_matches:
            img1_pts.append(kp1[match.queryIdx].pt)
            img2_pts.append(kp2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        # Compute homography matrix
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)  # x2 = M*x1  many times choose
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


def draw_matched_keypoint(img1, img2):
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
    cv2.imshow('match', outImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Main function definition
def main():
    # Get input set of images
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    # Equalize histogram
    img1 = equalize_histogram_color(img1)
    img2 = equalize_histogram_color(img2)

    # Use SIFT to find keypoints and return homography matrix
    M, img1_key, img2_key = get_sift_homography(img1, img2)
    # Show input images
    input_images = np.hstack((img1_key, img2_key))
    cv2.imshow('Input Images', input_images)

    # Stitch the images together using homography matrix
    result_image = get_stitched_image(img2, img1, M)

    # Write the result to the same directory
    result_image_name = 'results/'+'result_' + sys.argv[1][7:]
    cv2.imwrite(result_image_name, result_image)

    # Show the resulting image
    cv2.imshow('Result', result_image)
    draw_matched_keypoint(img1, img2)
    cv2.waitKey()


# Call main function
if __name__ == '__main__':
    main()
