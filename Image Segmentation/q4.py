import cv2 as cv
import numpy as np
from skimage.segmentation import felzenszwalb
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans


def create_dataset(hsv_image,rgb_image, number_of_features, number_of_labels):
    data_set = np.zeros((number_of_labels, number_of_features))
    for label in range(number_of_labels):
        indexes = np.where(segment_index_matrix == label)
        mean_h = np.mean(hsv_image[indexes[0], indexes[1], 0])
        mean_s = np.mean(hsv_image[indexes[0], indexes[1], 1])
        mean_v = np.mean(hsv_image[indexes[0], indexes[1], 2])

        mean_b = np.mean(rgb_image[indexes[0], indexes[1], 0])
        mean_g= np.mean(rgb_image[indexes[0], indexes[1], 1])
        mean_r = np.mean(rgb_image[indexes[0], indexes[1], 2])
        data_set[label, :] = np.array([mean_h, mean_s, mean_v, mean_b, mean_g, mean_r])
        # size_of_segment=np.size(indexes[0])
        # # data_set[label, :] = np.array([mean_h, mean_s, mean_v, mean_b, mean_g,mean_r,size_of_segment])
    return data_set




start_time=time.time()
image = cv.imread("birds.jpg")
image = cv.resize(image, (0, 0), fx=0.5, fy=0.5)
hsv_image=cv.cvtColor(image,cv.COLOR_BGR2HSV)
# lab=cv.cvtColor(image,cv.COLOR_BGR2Lab)
# luv=cv.cvtColor(image,cv.COLOR_BGR2Luv)
segment_index_matrix = felzenszwalb(image, scale=400, sigma=0.5, min_size=200)

dataSet=create_dataset(hsv_image,image, 6, np.max(segment_index_matrix) + 1)
bandwidth=20.5

ms=MeanShift(bandwidth=bandwidth)
ms.fit(dataSet)



#K-means
# dataset_copy = np.copy(dataSet)
# segment_index_matrix_copy = np.copy(segment_index_matrix)
# # km = KMeans(n_clusters=12,init='random')
# km = KMeans(n_clusters=14)
# km.fit_predict(dataset_copy)
# bird = (965, 1253)
# segment_of_bird = segment_index_matrix_copy[bird[0], bird[1]]
# cluster_index_of_bird = km.labels_[segment_of_bird]
# segment_numbers_of_birds = np.where(km.labels_ == cluster_index_of_bird)
# result = np.zeros_like(image)
#--------------------------------





bird=(965,1253)
segment_of_bird=segment_index_matrix[bird[0], bird[1]]
cluster_index_of_bird=ms.labels_[segment_of_bird]
segment_numbers_of_birds=np.where(ms.labels_==cluster_index_of_bird)
binary_image=np.zeros((image.shape[0],image.shape[1]))
for i in range(np.size(segment_numbers_of_birds)):
    index=np.where(segment_index_matrix==segment_numbers_of_birds[0][i])
    if np.size(index[0])>650:
        image[index[0],index[1],2]=255
        # binary_image[index[0], index[1]] = 1


# kernel = np.ones((3,3),np.uint8)
# cv.imwrite("Binary_image0.jpg",binary_image*255)
# binary_image=cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel,iterations=2)
# binary_image = cv.erode(binary_image,kernel,iterations = 2)
# cv.imwrite("Binary_image.jpg",binary_image*255)
# binary_image = cv.dilate(binary_image,kernel,iterations = 1)
# cv.imwrite("Binary_image2.jpg",binary_image*255)
# plt.imshow(binary_image,cmap='gray')
# plt.show()
# binary_image=np.stack((binary_image,np.copy(binary_image),np.copy(binary_image)),axis=2)
image = cv.resize(image, (0, 0), fx=2, fy=2)
cv.imwrite(f"res10.jpg",image)
print(time.time()-start_time)




