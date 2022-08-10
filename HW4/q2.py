import math
import random

import cv2 as cv
import numpy as np
import time



def distance(point1, point2):
    x = point1[0] - point2[0]
    x = x ** 2
    y = point1[1] - point2[1]
    y = y ** 2
    z = point1[2] - point2[2]
    z = z ** 2
    return math.sqrt(x + y + z)


def mean_shift(dataset, window_size, min_distance):
    labels = np.zeros((dataset.shape[0], 1))
    centroids = np.zeros((dataset.shape[0], dataset.shape[1] + 1))  # last column for label
    new_labeled_dataset = np.ones((dataset.shape[0], dataset.shape[1] + 1))
    new_labeled_dataset[:, :dataset.shape[1]] = np.copy(dataset)
    clustered = np.zeros((dataset.shape[0]))  # zero is False and one is True
    labelNumber = 0  # labels are 1,2,.....

    while True:
        new_dataset = np.copy(dataset).astype('float64')
        indexes = np.where(clustered == 0)
        # print(np.size(indexes[0]))
        if np.size(indexes[0]) == 0:
            break
        randIndx = random.randint(0, np.size(indexes[0]) - 1)
        current_centroid = np.copy(dataset[indexes[0][randIndx], :])
        clustered[indexes[0][randIndx]] = 1
        stop = False
        # in_window_points_X=[]
        while not stop:
            # print("+")
            distance_to_current_centroid = new_dataset[:, :] - current_centroid
            distance_to_current_centroid = distance_to_current_centroid * distance_to_current_centroid
            distance_to_current_centroid = np.sum(distance_to_current_centroid, axis=1)
            distance_to_current_centroid = np.sqrt(distance_to_current_centroid)
            in_window_points_indexes = np.where(distance_to_current_centroid < window_size)

            # x_indx=in_window_points_indexes[0]
            # x_indx=x_indx.tolist()
            # in_window_points_X=list(set(x_indx)| set(in_window_points_X))

            in_window_points = np.copy(dataset[in_window_points_indexes[0], :])
            x = np.mean(in_window_points[:, 0])
            y = np.mean(in_window_points[:, 1])
            z = np.mean(in_window_points[:, 2])
            new_centroid = np.array([x, y, z])
            if distance(current_centroid, new_centroid) > min_distance:
                # print("Still continue")
                current_centroid = new_centroid
            else:
                stop = True
                # in_window_points_X=np.array(in_window_points_X)
                # print(np.size(in_window_points_indexes[0]))
                clustered[in_window_points_indexes[0]] = 1
                if labelNumber != 0:
                    copy_centroids = np.copy(centroids[:labelNumber, :centroids.shape[1] - 1])
                    distance_of_new_centroid_to_other_centroids = copy_centroids - new_centroid
                    distance_of_new_centroid_to_other_centroids = distance_of_new_centroid_to_other_centroids ** 2
                    distance_of_new_centroid_to_other_centroids = np.sum(distance_of_new_centroid_to_other_centroids,
                                                                         axis=1)
                    distance_of_new_centroid_to_other_centroids = np.sqrt(distance_of_new_centroid_to_other_centroids)
                    near_centroid_indexes = np.where(distance_of_new_centroid_to_other_centroids <= min_distance)

                    if np.size(near_centroid_indexes[0]) > 0:
                        labels[in_window_points_indexes[0]] = centroids[
                            near_centroid_indexes[0][0], centroids.shape[1] - 1]
                    else:
                        centroids[labelNumber, :] = np.array(
                            [new_centroid[0], new_centroid[1], new_centroid[2], labelNumber + 1])
                        labelNumber += 1
                        labels[in_window_points_indexes[0]] = labelNumber
                else:
                    centroids[labelNumber, :] = np.array(
                        [new_centroid[0], new_centroid[1], new_centroid[2], labelNumber + 1])
                    labelNumber += 1
                    labels[in_window_points_indexes[0]] = labelNumber
                centroids[labelNumber, :] = np.array(
                    [new_centroid[0], new_centroid[1], new_centroid[2], labelNumber + 1])
                labelNumber += 1
                labels[in_window_points_indexes[0]] = labelNumber

    return labels, centroids, labelNumber


start_time = time.time()
image = cv.imread('park.jpg')
image = cv.resize(image, (0, 0), fx=0.25, fy=0.25)
image = cv.medianBlur(image, 7)
dataSet = image.reshape((image.shape[0] * image.shape[1], 3))
dataSet = dataSet.astype('float64')
labels, centroids, labelNumber = mean_shift(dataSet, 30, 20)

for i in range(labelNumber):
    index = np.where(labels == i + 1)
    if (np.size(index[0])) > 0:
        dataSet[index[0]] = np.array(
            [np.mean(dataSet[index[0], 0]), np.mean(dataSet[index[0], 1]), np.mean(dataSet[index[0], 2])])

dataSet = dataSet.reshape((image.shape[0], image.shape[1], 3))
dataSet = dataSet.astype('uint8')
dataSet = cv.resize(dataSet, (0, 0), fx=4, fy=4)
dataSet = cv.medianBlur(dataSet, 7)
cv.imwrite("res05.jpg", dataSet)
print(time.time() - start_time)
