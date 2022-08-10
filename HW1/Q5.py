import cv2 as cv2
import numpy as np
import time





def Good_convolution(image, size):
    k = (size - 1) // 2
    m = image.shape[0]
    n = image.shape[1]
    image = image / 9
    result = np.zeros((m - 2 * k, n - 2 * k))
    for i in range(size):
        for j in range(size):
            matrix = image[j:m - 2 * k + j, i:n - 2 * k + i]
            result = np.add(result, matrix)
    return result


def bad_convolution_box_filter(image, size):
    k = (size - 1) // 2
    k2 = np.power(size, 2)
    result = np.zeros((image.shape[0] - 2 * k, image.shape[1] - 2 * k), dtype='float32')
    image = image / k2
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            a = image[i:i + size, j:j + size]
            result[i, j] = np.sum(a)

    return result


size = 3
k = 1
start_time = time.time()
image = cv2.imread("Pink.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
opencvBlur = cv2.blur(image, (size, size))
cv2.imwrite("res07.jpg", opencvBlur)
print(f'Opencv Blur: {time.time() - start_time}')

# Bad Convolution
start_time = time.time()
bad_convolution = np.zeros((image.shape[0] - 2 * k, image.shape[1] - 2 * k, 3), dtype='uint8')
image = image / 255
bad_convolution[:, :, 0] = (bad_convolution_box_filter(image[:, :, 0], size) * 255).round().astype('uint8')
bad_convolution[:, :, 1] = (bad_convolution_box_filter(image[:, :, 1], size) * 255).round().astype('uint8')
bad_convolution[:, :, 2] = (bad_convolution_box_filter(image[:, :, 2], size) * 255).round().astype('uint8')
image = image * 255
cv2.imwrite("res08.jpg", bad_convolution)
print(f'Bad Convolution: {time.time() - start_time}')

# Good Convolution
start_time = time.time()
good_convolution = np.zeros((image.shape[0] - 2 * k, image.shape[1] - 2 * k, 3), dtype='uint8')
image = image / 255
good_convolution[:, :, 0] = (Good_convolution(image[:, :, 0], size) * 255).round().astype('uint8')
good_convolution[:, :, 1] = (Good_convolution(image[:, :, 1], size) * 255).round().astype('uint8')
good_convolution[:, :, 2] = (Good_convolution(image[:, :, 2], size) * 255).round().astype('uint8')
image = image * 255
cv2.imwrite("res09.jpg", good_convolution)
print(f'Good Convolution: {time.time() - start_time}')

cropped = opencvBlur[k:opencvBlur.shape[0] - k, k:opencvBlur.shape[1] - k, :]
print(f'number of pixels that Bad convolution is not equal to  opencv : {np.sum(np.not_equal(bad_convolution, cropped))}')
print(f'number of pixels that Good convolution is not equal to  opencv {np.sum(np.not_equal(good_convolution, cropped))}')
