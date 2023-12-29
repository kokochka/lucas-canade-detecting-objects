import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.signal import convolve2d

def lucas_kanade(imfirst, imsecond):
    myframe = cv.imread(imfirst)
    Image1 = cv.cvtColor(myframe, cv.COLOR_BGR2GRAY)
    newframe = cv.imread(imsecond)
    Image2 = cv.cvtColor(newframe, cv.COLOR_BGR2GRAY)

    color = np.random.randint(0, 255, (100, 3))
    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))

    Ix = (convolve2d(Image1, Gx) + convolve2d(Image2,Gx)) / 2
    Iy = (convolve2d(Image1, Gy) + convolve2d(Image2, Gy)) / 2
    It1 = convolve2d(Image1, Gt1) + convolve2d(Image2, Gt2)

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    features = cv.goodFeaturesToTrack(Image1, mask=None, **feature_params)
    feature = np.int32(features)
    feature = np.reshape(feature, newshape = [-1,2])

    u = np.ones(Ix.shape)
    v = np.ones(Ix.shape)
    status = np.zeros(feature.shape[0])
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    mask = np.zeros_like(myframe)

    newFeature = np.zeros_like(feature)

    for a, i in enumerate(feature):
        x, y = i
        A[0, 0] = np.sum((Ix[y - 1:y + 2, x - 1:x + 2]) ** 2)
        A[1, 1] = np.sum((Iy[y - 1:y + 2, x - 1:x + 2]) ** 2)
        A[0, 1] = np.sum(Ix[y - 1: y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
        A[1, 0] = np.sum(Ix[y - 1: y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
        Ainv = np.linalg.pinv(A)

        B[0, 0] = -np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
        B[1, 0] = -np.sum(Iy[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
        prod = np.matmul(Ainv, B)

        u[y, x] = prod[0]
        v[y, x] = prod[1]

        newFeature[a] = [np.int32(x + u[y,x]), np.int32(y +v[y, x])]
        if np.int32(x + u[y, x]) == x and np.int32(y + v[y, x]) == y:
            status[a] = 0
        else:
            status[a] = 1
    um = np.flipud(u)
    vm = np.flipud(v)

    good_new = newFeature[status==1]
    good_old = feature[status==1]
    print(good_new.shape)
    print(good_old.shape)

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        newframe = cv.circle(newframe, (a, b), 5, color[i].tolist(), -1)
    img = cv.add(newframe, mask)
    return img

test = lucas_kanade("snap1.png", "snap3.png")

basketball_image = cv.cvtColor(test, cv.COLOR_BGR2RGB)
plt.imshow(test)
plt.show()