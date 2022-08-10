import math

import cv2 as cv
import numpy as np
from skimage.segmentation import mark_boundaries
import time


def q3(image_name, k, result_name, color):
    # circle_size = 10
    start_time = time.time()
    image = cv.imread(image_name)
    image = cv.resize(image, (0, 0), image, 0.1, 0.1, cv.INTER_AREA)
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    image = image.astype('float64')
    lab_image = lab_image.astype('float64')
    lab_image = cv.GaussianBlur(lab_image, (7, 7), 2)
    height = image.shape[0]
    width = image.shape[1]
    s = math.sqrt((width * height) / k)
    s_divide_2 = math.floor(s / 2)
    alpha = 10 / s
    cluster_centers = np.zeros((2 * k, 3))  # last column is cluster name
    # size is 2*k because as mentioned in doc we could have more than k cluster center
    counter = 0
    print(height)
    print(width)
    print(s)
    copy_image = np.copy(image)
    for i in range(s_divide_2, height, math.floor(s)):
        for j in range(s_divide_2, width, math.floor(s)):
            cluster_centers[counter, :] = np.array([i, j, counter])
            # cv.circle(copy_image, (j, i), circle_size, (0, 0, 255), -1)
            counter += 1

    # cv.imwrite(("Cluster_centers.jpg"),copy_image)
    # print(counter)

    cluster_centers = cluster_centers[:counter, :]
    number_of_clusters = cluster_centers.shape[0]

    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    gradient_image = sobel_x ** 2 + sobel_y ** 2
    # print(type(gradient_image[0,0,0]))
    gradient_image = np.sqrt(gradient_image)
    gradient_image = np.mean(gradient_image, axis=2)
    # cv.imwrite(f"Gradient{k}.jpg", gradient_image)
    # print(gradient_image.shape)

    copy2_image = np.copy(image)
    for i in range(number_of_clusters):
        center = np.copy(cluster_centers[i, :])
        center = center.astype('int64')
        up_index = max(0, center[0] - 2)
        down_index = min(height, center[0] + 3)
        right_index = min(width, center[1] + 3)
        left_index = max(0, center[1] - 2)
        min_i = 0
        min_j = 0
        min_val = math.inf
        for i_index in range(up_index, down_index):
            for j_index in range(left_index, right_index):
                if gradient_image[i_index, j_index] < min_val:
                    min_val = gradient_image[i_index, j_index]
                    min_i = i_index
                    min_j = j_index
        cluster_centers[i, :] = np.array([min_i, min_j, cluster_centers[i, 2]])
        # cv.circle(copy2_image, (int(cluster_centers[i, 1]), int(cluster_centers[i, 0])), circle_size, (0, 0, 255), -1)

    # print(cluster_centers)
    # cv.imwrite(f"new_Cluster_center{k}.jpg", copy2_image)
    # exit(0)

    labels = np.zeros((lab_image.shape[0], lab_image.shape[1]))
    distance = np.ones((lab_image.shape[0], lab_image.shape[1]))
    distance_label = np.ones((lab_image.shape[0], lab_image.shape[1]))
    distance = distance * math.inf
    distance_label = distance_label * -1
    max_iter = 5
    iteration = 0

    # stop = False
    x_indx = np.arange(height)
    y_indx = np.arange(width)
    x_indx = x_indx.reshape(height, 1)
    y_indx = y_indx.reshape(1, width)
    x_indx = np.zeros((height, width)) + x_indx
    y_indx = np.zeros((height, width)) + y_indx

    while iteration < max_iter:
        for i in range(number_of_clusters):
            up_index = max(0, math.floor(cluster_centers[i, 0] - s))
            down_index = min(lab_image.shape[0], math.floor(cluster_centers[i, 0] + s))
            right_index = min(lab_image.shape[1], math.floor(cluster_centers[i, 1] + s))
            left_index = max(0, math.floor(cluster_centers[i, 1] - s))
            local_lab_image = np.copy(lab_image[up_index:down_index, left_index:right_index])
            local_LAB_distance_to_ith_cluster_center = np.zeros_like(local_lab_image)
            local_LAB_distance_to_ith_cluster_center = local_lab_image - np.copy(lab_image[int(cluster_centers[i, 0]),
                                                                                 int(cluster_centers[i, 1]),
                                                                                 :])

            local_LAB_distance_to_ith_cluster_center = local_LAB_distance_to_ith_cluster_center ** 2
            local_LAB_distance_to_ith_cluster_center = np.sum(local_LAB_distance_to_ith_cluster_center, axis=2)
            local_LAB_distance_to_ith_cluster_center = np.sqrt(local_LAB_distance_to_ith_cluster_center)
            x_distance = x_indx[up_index:down_index, left_index:right_index] - cluster_centers[i, 0]
            x_distance = x_distance ** 2
            y_distance = y_indx[up_index:down_index, left_index:right_index] - cluster_centers[i, 1]
            y_distance = y_distance ** 2
            xy_distance = x_distance + y_distance
            xy_distance = np.sqrt(xy_distance)
            total_distance = local_LAB_distance_to_ith_cluster_center + alpha * xy_distance
            indexes = np.where(distance[up_index:down_index, left_index:right_index] > total_distance)
            labels[up_index:down_index, left_index:right_index][indexes[0], indexes[1]] = i
            distance[up_index:down_index, left_index:right_index][indexes[0], indexes[1]] = total_distance[
                indexes[0], indexes[1]]

        #####
        for i in range(number_of_clusters):
            indexes = np.where(labels == i)
            x_mean = np.mean(indexes[0])
            y_mean = np.mean(indexes[1])
            if np.size(indexes[0] > 0):
                x_mean = math.floor(x_mean)
                y_mean = math.floor(y_mean)
                new_cluster = np.array([x_mean, y_mean])
                cluster_centers[i, :cluster_centers.shape[1] - 1] = new_cluster
        iteration += 1
        print(f"End iteration {iteration - 1}")

    labels = labels.astype('int64')
    image = mark_boundaries(image, labels, color=color)
    image = cv.resize(image, (0, 0), image, 10, 10, cv.INTER_AREA)
    cv.imwrite(result_name, image)
    print(f'{k} segment: {time.time() - start_time} second')


begin_time = time.time()
q3("slic.jpg", 64, 'res06.jpg', (0, 0, 0))
q3("slic.jpg", 256, 'res07.jpg', (180, 105, 255))
q3("slic.jpg", 1024, 'res08.jpg', (0, 0, 0))
q3("slic.jpg", 2048, 'res09.jpg', (0, 0, 255))
print(f'Total time: {time.time() - begin_time}')
