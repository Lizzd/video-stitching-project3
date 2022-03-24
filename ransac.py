import cv2

from Homography import Homography
import numpy as np


def ransac(img1_pts, img2_pts, epsilon, n):
    all_index = np.arange(img1_pts.shape[0])
    max_inlier = 0
    best_H = []
    best_inlier = []
    for i in range(n):
        # step1: randomly choose 4 samples
        index = np.random.choice(all_index, 4)
        test_img1_pts = img1_pts[index]
        test_img2_pts = img2_pts[index]

        # step2: fit the model to the random samples
        # use the 4 samples to find the H
        samples = np.zeros((4, 6))
        for i in range(4):
            samples[i] = np.stack((test_img1_pts[i], test_img2_pts[i])).flatten()
        H = Homography(samples)

        transf_pts = []
        for i in range(len(img1_pts)):
            transf_pt = H * img1_pts[i].T
            transf_pts.append(transf_pt.T)
        transf_pts = np.float32(transf_pts)
        # step3: count the inliers
        num_of_inlier = 0
        inlier = []
        for j in range(img1_pts.shape[0]):
            F = cv2.findFundamentalMat()
            dist = np.linalg.norm(img2_pts[j] - transf_pts[j])
            if dist <= epsilon:
                num_of_inlier += 1
                inlier.append(np.stack((img1_pts[j], img2_pts[j])).flatten())

        if num_of_inlier >= max_inlier:
            max_inlier = num_of_inlier
            # store the matrix
            best_H = H
            best_inlier = inlier

    # choose the model that has the maximum number of inliers
    # recompute the H using all the inliers
    return best_inlier
