import cv2
import random

from Homography import Homography
import numpy as np
from FundMatrix import epa

# def ransac(img1_pts, img2_pts, epsilon, n):
#     all_index = np.arange(img1_pts.shape[0])

#     max_inlier = 0
#     best_H = []
#     best_inlier = []
#     for i in range(n):
#         # step1: randomly choose 4 samples
#         index = np.random.choice(all_index, 4)
#         test_img1_pts = img1_pts[index]
#         test_img2_pts = img2_pts[index]

#         # step2: fit the model to the random samples
#         # use the 4 samples to find the H
#         samples = np.zeros((4, 6))
#         samples[:, 2] = 1
#         samples[:, 5] = 1
#         for i in range(4):
#             tmp = np.stack((test_img1_pts[i], test_img2_pts[i])).flatten()
#             # print("tmp", tmp)
#             samples[i] = tmp

#         if len(samples) == 0:
#             print("sample null")
#         H = Homography(samples)
#         F = epa(samples)

#         transf_pts = []
#         for i in range(len(img1_pts)):
#             transf_pt = H @ img1_pts[i].T
#             transf_pts.append(transf_pt.T)
#         transf_pts = np.float32(transf_pts)
#         # step3: count the inliers
#         num_of_inlier = 0
#         inlier = []
#         for j in range(img1_pts.shape[0]):
#             # dist = (img2_pts[j] - transf_pts[j]).sum()**2
#             dist = cv2.sampsonDistance(img2_pts[j], transf_pts[j], F)
#             if dist <= epsilon:
#                 num_of_inlier += 1
#                 inlier.append(np.stack((img1_pts[j], img2_pts[j])).flatten())
#                 print("done")

#         if num_of_inlier >= max_inlier:
#             max_inlier = num_of_inlier
#             # store the matrix
#             best_H = H
#             best_inlier = inlier

#     # choose the model that has the maximum number of inliers
#     # recompute the H using all the inliers
#     return best_inlier


def calculateHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h


#
#Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


#
#Runs through ransac algorithm, creating homographies from random correspondences
#
def ransac(corr):
    maxInliers = []
    finalH = None
    for i in range(1000):
        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        print("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

    return finalH, maxInliers