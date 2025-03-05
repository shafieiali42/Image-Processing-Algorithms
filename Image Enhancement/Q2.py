import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def show_hist(histogram, edge_bins, title, xLim, xLabel, yLabel):
    plt.figure()
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim(xLim)
    plt.plot(edge_bins[0:-1], histogram)
    plt.show()


start_time = time.time()
image = cv2.imread("Enhance2.JPG")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
v_hist, edge_bins = np.histogram(hsv[:, :, 2], 256, range=(0, 256), density=True)
# show_hist(v_hist, edge_bins, "V Histogram Before change", [0, 256], "Intensity", "probability")
s = np.cumsum(v_hist)
# show_hist(s, edge_bins, "V Histogram", [0, 256], "Intensity", "probability")
s = s * 255
s=np.round(s)
hsv[:, :, 2] = s[hsv[:, :, 2]]

image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
print(time.time() - start_time)
v_hist, edge_bins = np.histogram(hsv[:, :, 2], 256, range=(0, 256), density=True)
# show_hist(v_hist, edge_bins, "Equalized V Histogram", [0, 256], "Intensity", "probability")
cv2.imwrite("res02.jpg", image)
