import cv2 as cv
import numpy as np
import time


def gaussian_func(x, y, sd):
    result = (1 / (2 * np.pi * np.power(sd, 2))) * np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sd, 2)))
    return result


def gaussian_filter(size):
    sd = np.floor(size / 3)
    filter = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            filter[i, j] = gaussian_func(i, j, sd)
    filter = filter * (1 / np.sum(filter))
    return filter


def gaussian_blur(image, size,zipped):
    blurred_image = convolution(image, gaussian_filter(size), size,zipped)
    return blurred_image


def convolution(image, filter, size,zipped):
    start = time.time()
    k = (size - 1) // 2
    m = image.shape[0]
    n = image.shape[1]
    result = np.zeros((m - 2 * k, n - 2 * k))
    for i in range(size):
        for j in range(size):
            matrix = filter[i, j] * image[j:m - 2 * k + j, i:n - 2 * k + i]
            result = result + matrix

    # print(f' Convolution Time.....................: {time.time() - start}')
    # print(f'Result size: {result.shape}')
    zipped = zipped[~np.any(((zipped < k) | (zipped>= image.shape[0]-k)| (zipped>= image.shape[1]-k)), axis=1)]
    result[(zipped[:,0]-k),(zipped[:,1]-k)]=image[zipped[:,0],zipped[:,1]]
    return result


size = 9
k = 4
print("start")
start_time = time.time()
image = cv.imread("Flowers.jpg")

hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
a=np.where((hsv[:, :, 0] <= 165) & (hsv[:, :, 0] >= 140) & (hsv[:, :, 1] >= 20) & (hsv[:, :, 1] <= 255) &
    (hsv[:, :, 2] >= 100) & (hsv[:, :, 2] <= 255))
condition=((hsv[:, :, 0] <= 165) & (hsv[:, :, 0] >= 140) & (hsv[:, :, 1] >= 20) & (hsv[:, :, 1] <= 255) &
    (hsv[:, :, 2] >= 100) & (hsv[:, :, 2] <= 255))
hsv[a[0],a[1],0]=28
image=cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
zipped=np.array(list(zip(*np.where(condition))))

image = image / 255
r = (gaussian_blur(image[:, :, 2], size,zipped) * 255).round().astype('uint8')
g = (gaussian_blur(image[:, :, 1], size,zipped) * 255).round().astype('uint8')
b = (gaussian_blur(image[:, :, 0], size,zipped) * 255).round().astype('uint8')
image = image * 255
my_blurred = cv.merge([b, g, r])

cv.imwrite("res06.jpg", my_blurred)
print("end")
end_time = time.time()
print(end_time - start_time)
cv.waitKey()
