import math
import numpy as np
import matplotlib.pyplot as plt
import random


def k_Means(dataSet, k):
    new_labeled_dataset = np.zeros((dataSet.shape[0], dataSet.shape[1] + 1))
    new_labeled_dataset[:, :dataSet.shape[1]] = np.copy(dataSet)
    random_point_index = random.sample(range(0, dataSet.shape[0]), k)
    current_center_points = [point for point in dataSet[random_point_index, :]]
    current_center_points = np.array(current_center_points, dtype='float64')
    last_step_center_point = np.zeros_like(current_center_points)
    point_label = np.zeros((dataSet.shape[0], 1))
    distance = np.zeros((dataSet.shape[0], k))
    stop = False
    counter = 0
    while not stop:
        if counter > 0:
            for i in range(k):
                labels = point_label
                indexes = np.where(labels == i)
                if np.size(indexes) > 0:
                    x = np.mean(new_labeled_dataset[indexes, 0])
                    y = np.mean(new_labeled_dataset[indexes, 1])
                    current_center_points[i] = np.array([x, y])
                    if np.sum(np.not_equal(current_center_points, last_step_center_point)) ==0:
                        stop = True
                        break
        if not stop:
            for i in range(k):
                new_dataset = np.copy(new_labeled_dataset).astype('float64')
                distance_to_ith_point = new_dataset[:, :new_dataset.shape[1] - 1] - current_center_points[i, :]
                distance_to_ith_point = distance_to_ith_point * distance_to_ith_point
                distance_to_ith_point = np.sum(distance_to_ith_point, axis=1)
                distance_to_ith_point = distance_to_ith_point
                distance[:, i] = distance_to_ith_point

            point_label = np.argmin(distance, axis=1)
            # print(point_label)
            new_labeled_dataset[:, new_labeled_dataset.shape[1] - 1] = point_label
            last_step_center_point = np.copy(current_center_points)
            counter += 1

    # a = current_center_points[:, 0]
    # b = current_center_points[:, 1]
    # plt.scatter(a, b, color='black', s=100)
    # plt.scatter(np.array([10,2,3]),np.array([-10,-1,-1]),color='black')
    return point_label


file = open("Points.txt")
dataSet = file.readlines()
number_of_points = int(dataSet[0])
dataSet = [[float(val) for val in line.split()] for line in dataSet[1:]]
dataSet = np.array(dataSet)
print(dataSet.shape)

plt.scatter(dataSet[:, 0], dataSet[:, 1])
plt.title("Points")
plt.xlabel("X")
plt.ylabel("Y")
# plt.show()
plt.savefig('res01.jpg')
plt.close()




for i in range(2):
    k = 2
    labels = k_Means(dataSet, k)
    colors = ['red', 'blue', 'green', 'orange']
    plt.title("Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    for i in range(k):
        indexes = np.where(labels == i)
        plt.scatter(dataSet[indexes, 0], dataSet[indexes, 1], color=colors[i])

    plt.savefig(f'res0{i+2}.jpg')
    plt.close()


def make_polar(point):
    r = point[0] ** 2 + point[1] ** 2
    r = math.sqrt(r)
    theta = math.atan(point[1] / point[0])
    return r, theta


def create_polar_dataset(dataSet):
    polar_dataSet = np.copy(dataSet)
    for i in range(dataSet.shape[0]):
        polar_dataSet[i, :] = make_polar(polar_dataSet[i, :])
    return polar_dataSet


k=2
polar_dataSet = create_polar_dataset(dataSet)
labels = k_Means(polar_dataSet, k)
colors = ['red', 'blue', 'green', 'orange']
plt.title("Points")
plt.xlabel("X")
plt.ylabel("Y")
for i in range(k):
    indexes = np.where(labels == i)
    plt.scatter(dataSet[indexes, 0], dataSet[indexes, 1], color=colors[i])

plt.savefig('res04.jpg')
plt.close()

# plt.title("Points")
# plt.xlabel("r")
# plt.ylabel("theta")
# for i in range(k):
#     indexes = np.where(labels == i)
#     plt.scatter(polar_dataSet[indexes, 0], polar_dataSet[indexes, 1], color=colors[i])
#
# plt.savefig('result.jpg')
