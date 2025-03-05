import cv2 as cv
import numpy as np
from scipy import signal




def ncc(image,template):
    template = template.astype('float64')
    image = image.astype('float64')
    template = template - np.mean(template)
    ones = np.ones_like(template, dtype='float32')
    image_bar = signal.correlate(image, ones, mode='same')
    num = signal.correlate(image, template, mode='same') - np.sum(template) * image_bar / np.size(template)
    template2 = template ** 2
    se_template = np.sqrt(np.sum(template2))
    image2 = image ** 2
    k = signal.correlate(image2, ones, mode='same')
    denominator = k - (image_bar ** 2) / np.size(template)
    denominator = np.sqrt(denominator)
    denominator = se_template * denominator
    return num / denominator


def template_matching(image,template,threshold):
    ncc_matrix=ncc(image,template)
    # ncc_image=np.copy(ncc_matrix)
    # ncc_image=cv.normalize(ncc_image, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
    locations=np.where(ncc_matrix>threshold)
    for loc in list(zip(*locations[::-1])):
        top = (int(loc[0]), int(loc[1]))
        pos = [top[0], top[1], template.shape[1], template.shape[0]]
        rectangels.append(pos)
        rectangels.append(pos)




rectangels=[]
ship=cv.imread('Greek-ship.jpg')
gray_ship=cv.cvtColor(ship,cv.COLOR_BGR2GRAY)
patch=cv.imread('patch.png')
gray_patch=cv.cvtColor(patch,cv.COLOR_BGR2GRAY)
gray_patch=gray_patch[50:-50,50:-50]
dx=0.5
gray_patch=cv.resize(gray_patch,(0,0),fx=dx,fy=dx)
template_matching(gray_ship,gray_patch,0.47)
rectangels,weights=cv.groupRectangles(rectangels,2,0.15)
w_patch=gray_patch.shape[1]
h_patch=patch.shape[0]
# print(w_patch)
i=0
color=[(0,255,255),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(255,0,0)]
for (x,y,w,h) in rectangels:
    cv.rectangle(ship,(x-w_patch//2,y-h_patch//2),(x+w_patch//2,y+h_patch//2),color[i%6],2)
    i=i+1

cv.imwrite("res15.jpg",ship)