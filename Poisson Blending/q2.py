import cv2 as cv
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve



def add(target,sourceName,maskName,target_point,result):
    source = cv.imread(f"{sourceName}")
    src_mask = cv.imread(f"{maskName}")
    # target_mask = src_mask
    source = source.astype("float64")
    target = target.astype("float64")

    src_mask = cv.cvtColor(src_mask, cv.COLOR_BGR2GRAY)
    target_top_left = target_point

    target_rect = target[target_top_left[0]:target_top_left[0] + source.shape[0],
                  target_top_left[1]:target_top_left[1] + source.shape[1]]

    src_laplacian = cv.Laplacian(source, cv.CV_64F)
    offset = np.array([0, -1, -target_rect.shape[1], 1,target_rect.shape[1]])
    src_mask_rect_index = np.where(src_mask > 0)
    indexes = src_mask_rect_index[0] * target_rect.shape[1]
    indexes=indexes+ src_mask_rect_index[1]
    size = np.size(target_rect) // 3
    diag0 = np.ones(size)
    diag0[indexes] = -4
    diag1 = np.zeros(size)
    diag1[indexes] = 1
    diag1 = np.roll(diag1, -1)
    diag2 = np.zeros(size)
    diag2[indexes] = 1
    diag2 = np.roll(diag2, -target_rect.shape[1])
    diag3 = np.zeros(size)
    diag3[indexes] = 1
    diag3 = np.roll(diag3,1)
    diag4 = np.zeros(size)
    diag4[indexes] = 1
    diag4 = np.roll(diag4, target_rect.shape[1])
    diagonals = np.array([diag0, diag1, diag2, diag3, diag4])
    a = sparse.dia_matrix((diagonals, offset), (size, size))
    result_rect = np.zeros_like(target_rect)

    for i in range(3):
        b = np.zeros(size)
        # print(size)
        laplacian_vector = src_laplacian[:, :, i].reshape(size)
        b = target_rect[:, :, i].reshape(size)
        b[indexes] = laplacian_vector[indexes]
        x = spsolve(a.tocsr(), b)
        x = x.reshape(target_rect.shape[0], target_rect.shape[1])
        result_rect[:, :, i] = x


    result[target_top_left[0]:target_top_left[0] + source.shape[0],
    target_top_left[1]:target_top_left[1] + source.shape[1]] = result_rect
    result = np.clip(result, 0, 255)
    # result*=255
    # print(np.max(result))
    # print(np.min(result))
    return result




target = cv.imread(f"res06.jpg")
result = target.copy()
result = result.astype('float64')
result=add(target,"res05.jpg","city_mask2.png",(306,0),result)

result = result.astype('uint8')
cv.imwrite("res07.jpg", result)