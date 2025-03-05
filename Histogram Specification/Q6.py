import cv2
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.exposure import match_histograms


def show_hist(histogram, edge_bins, title, xLim, xLabel, yLabel):
    plt.figure()
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim(xLim)
    plt.plot(edge_bins[0:-1], histogram)
    plt.savefig(f'{title}.jpg')
    plt.show()


def show_two_hist(histogram1, histogram2, edge_bins1, edge_bins2, title, xLim, xLabel, yLabel):
    plt.figure()
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim(xLim)
    plt.plot(edge_bins1[0:-1], histogram1)
    plt.plot(edge_bins2[0:-1], histogram2)
    plt.savefig(f'{title}.jpg')
    plt.show()

def show_three_hist(histogram1, histogram2,histogram3, edge_bins1, edge_bins2,edge_bins3, title, xLim, xLabel, yLabel,result_name):
    plt.figure()
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim(xLim)
    plt.plot(edge_bins1[0:-1], histogram1,color='red')
    plt.plot(edge_bins2[0:-1], histogram2,color='blue')
    plt.plot(edge_bins3[0:-1], histogram3,color='green')
    plt.savefig(f'{result_name}.jpg')
    plt.show()


def histogram_specification(image, image_hist, target_hist, chanel_number):
    s = np.cumsum(image_hist)
    s = s * 255
    s = np.round(s).astype('int32')
    m = np.arange(256)
    m = m * 0
    print(f'M: {m.shape}')
    print(f'S: {s.shape}')
    print(m[0])

    for i in range(256):
        if m[s[i]] == 0:
            m[s[i]] = i

    print(f"image: {image.shape}")
    image[:, :, chanel_number] = s[image[:, :, chanel_number]]
    g = np.cumsum(target_hist)
    g = g * 255
    s = np.reshape(s, (1, 256))
    g = np.reshape(g, (256, 1))
    print(s.shape)
    print(g.shape)
    diff = g - s
    diff = np.abs(diff)
    print(diff.shape)
    mins = np.argmin(diff, axis=0)
    print(f'mins:  {mins.shape}')
    g = np.reshape(g, (256,))
    image[:, :, chanel_number] = mins[m[image[:, :, chanel_number]]]
    return image


start_time = time.time()
flower_image = cv2.imread("Pink.jpg")
dark_image = cv2.imread("Dark.jpg")
dark_hsv = cv2.cvtColor(dark_image, cv2.COLOR_BGR2HSV)
flower_hsv = cv2.cvtColor(flower_image, cv2.COLOR_BGR2HSV)
#
# dark_image=match_histograms(dark_image,flower_image,multichannel=True)

flower_blue_hist, blue_flower_edge_bins = np.histogram(flower_image[:, :, 0], 256, range=(0, 256), density=True)
flower_blue_cum_hist = np.cumsum(flower_blue_hist)
# show_hist(flower_blue_hist, blue_flower_edge_bins, "Flower blue  chanel", [0, 256], "Intensity", "p")
flower_red_hist, red_flower_edge_bins = np.histogram(flower_image[:, :, 2], 256, range=(0, 256), density=True)
flower_red_cum_hist = np.cumsum(flower_red_hist)
# show_hist(flower_red_cum_hist, red_flower_edge_bins, "Flower red  chanel", [0, 256], "Intensity", "p")
flower_green_hist, green_flower_edge_bins = np.histogram(flower_image[:, :, 1], 256, range=(0, 256), density=True)
flower_green_cum_hist = np.cumsum(flower_green_hist)
# show_hist(flower_green_hist, green_flower_edge_bins, "Flower green  chanel", [0, 256], "Intensity", "p")


dark_blue_hist, blue_dark_edge_bins = np.histogram(dark_image[:, :, 0], 256, range=(0, 256), density=True)
# show_hist(dark_blue_hist, blue_dark_edge_bins, "Dark blue  chanel", [0, 256], "Intensity", "p")
dark_green_hist, green_dark_edge_bins = np.histogram(dark_image[:, :, 1], 256, range=(0, 256), density=True)
# show_hist(dark_green_hist, green_dark_edge_bins, "Dark green  chanel", [0, 256], "Intensity", "p")
dark_red_hist, red_dark_edge_bins = np.histogram(dark_image[:, :, 2], 256, range=(0, 256), density=True)
# show_hist(dark_red_hist, red_dark_edge_bins, "Dark red  chanel", [0, 256], "Intensity", "p")

dark_image = histogram_specification(dark_image, dark_blue_hist, flower_blue_hist, 0)
dark_image = histogram_specification(dark_image, dark_green_hist, flower_green_hist, 1)
dark_image = histogram_specification(dark_image, dark_red_hist, flower_red_hist, 2)

dark_blue_hist, blue_dark_edge_bins = np.histogram(dark_image[:, :, 0], 256, range=(0, 256), density=True)
dark_blue_cum_hist = np.cumsum(dark_blue_hist)
# show_hist(dark_blue_hist, blue_dark_edge_bins, "Dark blue  chanel", [0, 256], "Intensity", "p")

dark_green_hist, green_dark_edge_bins = np.histogram(dark_image[:, :, 1], 256, range=(0, 256), density=True)
dark_green_cum_hist = np.cumsum(dark_green_hist)
# show_hist(dark_green_hist, green_dark_edge_bins, "Dark green  chanel", [0, 256], "Intensity", "p")

dark_red_hist, red_dark_edge_bins = np.histogram(dark_image[:, :, 2], 256, range=(0, 256), density=True)
dark_red_cum_hist = np.cumsum(dark_red_hist)
# show_hist(dark_red_hist, red_dark_edge_bins, "Dark red  chanel", [0, 256], "Intensity", "p")
show_two_hist(dark_red_cum_hist, flower_red_cum_hist, red_dark_edge_bins, red_flower_edge_bins, "Red cumulative", [0, 256], "Intensity", "p")
# show_two_hist(dark_red_hist,flower_red_hist,red_dark_edge_bins,red_flower_edge_bins,"Red",[0,256],"Intensity","p")

show_two_hist(dark_blue_cum_hist, flower_blue_cum_hist, blue_dark_edge_bins, blue_flower_edge_bins, "Blue cumulative", [0, 256], "Intensity", "p")
# show_two_hist(dark_blue_hist,flower_blue_hist,blue_dark_edge_bins,blue_flower_edge_bins,"Blue",[0,256],"Intensity","p")

show_two_hist(dark_green_cum_hist, flower_green_cum_hist, green_dark_edge_bins, green_flower_edge_bins, "Green cumulative", [0, 256], "Intensity", "p")
# show_two_hist(dark_green_hist,flower_green_hist,green_dark_edge_bins,green_flower_edge_bins,"Green",[0,256],"Intensity","p")
# dark_v_hist, v_dark_edge_bins=np.histogram(dark_image[:, :, 0], 256, range=(0, 256), density=True)
# flower_v_hist, v_flower_edge_bins=np.histogram(dark_image[:, :, 0], 256, range=(0, 256), density=True)
# dark_hsv=histogram_specification(dark_hsv,dark_v_hist,flower_v_hist,2)
# dark_image=cv2.cvtColor(dark_hsv,cv2.COLOR_HSV2BGR)

show_three_hist(dark_red_hist,dark_blue_hist,dark_green_hist,red_dark_edge_bins,blue_dark_edge_bins,green_dark_edge_bins,
                "Dark Histogram after change",[0,256],'Intensity','P','res10')

show_three_hist(flower_red_hist,flower_blue_hist,flower_green_hist,red_flower_edge_bins,blue_flower_edge_bins,green_flower_edge_bins,
                "Flower Histogram",[0,256],'Intensity','P','Flower_hist')



print(time.time() - start_time)
cv2.imwrite("res11.jpg", dark_image)
