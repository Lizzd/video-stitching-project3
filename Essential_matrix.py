import numpy as np


def get_Homograph_mat(correspondences):
    correspondences_robust = F_ransac(correspondences, 'tolerance', 0.04)
    E = epa(correspondences_robust)


def epa(correspondences):
    size_x = np.shape(correspondences, 1)
    homo = np.ones(1, size_x)
    x1 = np.vstack((correspondences[0:1, :], homo))
    x2 = np.vstack((correspondences[2:3, :], homo))  ##如果K不给定那么x1代表的是相机坐标系中的坐标
    A = np.zeros(size_x, 9)
    for i in range(size_x):
        A[i, :] = np.kron(x1[:, i], x2[:, i])

        [_, _, V] = np.linalg.svd(A);  ## 最终我们需要的都是相机坐标系中的坐标
