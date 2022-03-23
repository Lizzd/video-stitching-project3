import numpy as np


def epa(correspondences):
    size_x = np.shape(correspondences, 1)
    homo = np.ones(1, size_x)
    x1 = np.vstack((correspondences[0:1, :], homo))
    x2 = np.vstack((correspondences[2:3, :], homo))
    A = np.zeros(size_x, 9)
    for i in range(size_x):
        A[i, :] = np.kron(x1[:, i], x2[:, i])

    [_, _, V] = np.linalg.svd(A)
    G = np.resize(V[:, 9], [3, 3])
    [U, S, V] = np.linalg.svd(G)
    FundMatrix = U * [[S[1, 1], 0, 0], [0, S[2, 2], 0], [0, 0, 0]] * V.T

    return FundMatrix