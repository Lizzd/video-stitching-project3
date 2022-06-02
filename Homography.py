import numpy as np


def Homography(correspondences):
    # loop through correspondences and create assemble matrix
    aList = []
    # countpairs = correspondences.shape[0]*2
    A_matrix = np.zeros((8, 9))
    if (len(correspondences) == 0):
        print("correspondences null", correspondences)
    for i, corr in enumerate(correspondences):
        # p1 = np.matrix([corr.item(0), corr.item(1), 1])
        # p2 = np.matrix([corr.item(2), corr.item(3), 1])
        #
        # a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
        #       p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        # a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
        #       p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        x1 = corr[0]
        y1 = corr[1]
        x11 = corr[2]
        y11 = corr[3]

        a1 = [x1, y1, 1, 0, 0, 0, -x1 * x11, -y1 * x11, -x11]
        a2 = [0, 0, 0, x1, y1, 1, -x1 * y11, -y1 * y11, -y1]
        aList.append(a1)
        aList.append(a2)
        # A_matrix[i,:] = a1;
        # A_matrix[i+1,:] = a2;
    if (len(aList)<2):
        print("alist null")

    for j in range(0, 8, 2):
        A_matrix[j, :] = aList[j]
        A_matrix[j + 1, :] = aList[j + 1]
    # matrixA = np.mat(aList)

    # svd composition
    A_matrix = A_matrix.T@A_matrix
    u, s, v = np.linalg.svd(A_matrix)

    # reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    # normalize and now we have h
    h = (1 / h.item(8)) * h
    return h
