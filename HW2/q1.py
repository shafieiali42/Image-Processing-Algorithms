import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal





def show_gaussain_filter(size,sigma):
    gaussian_x=cv.getGaussianKernel(size,sigma)
    gaussian_y=cv.getGaussianKernel(size,sigma)
    gaussian_y=np.transpose(gaussian_y)
    gaussian=gaussian_x*gaussian_y
    cv.imwrite('res01.jpg',gaussian)
    return gaussian


def LoG(sigma,x,y):
    e=np.exp(-(x**2+y**2)/(2*(sigma**2)))
    a=(x**2+y**2-2*(sigma**2))/(2*np.pi*(sigma**6))
    return a*e



def Log_filter(sigma,size):
    k=size//2
    filter=np.zeros((size,size),dtype='float64')
    for i in range(-k,k+1):
        for j in range(-k,k+1):
            filter[i+k,j+k]=LoG(sigma,np.abs(i),np.abs(j))
    
    return filter

def show_magnitude(image,chanelName):
    image2=np.copy(image)
    image2=image2.astype('float64')
    image_fft=np.fft.fft2(image2)
    fft_shifted=np.fft.fftshift(image_fft)
    amplitude_image=np.abs(fft_shifted)
    log_image=np.log(amplitude_image)
    plt.imsave(f"{chanelName}res08.jpg",log_image,cmap='gray')
   

def normalize(image):
    image =image- image.mean()
    image = image/image.std()
    image = image*30
    image = image+127
    return image

def rgb_sharping(image,sigma,k):
    blue=sharp_chanel(image[:,:,0],sigma,k,'blue')
    green=sharp_chanel(image[:,:,1],sigma,k,'green')
    red=sharp_chanel(image[:,:,2],sigma,k,'red')
    image[:,:,0]=blue
    image[:,:,1]=green
    image[:,:,2]=red
    image= np.clip(image, 0, 255).astype('uint8')
    return image

def sharp_chanel(image,sigma,k,chanelName):
    image=image.astype('float64')
    image_fft=np.fft.fft2(image)
    shifted_fft=np.fft.fftshift(image_fft)
    hp=Highpass_filter(sigma,image.shape[1],image.shape[0])
    x=1+k*hp
    y=shifted_fft*x
    y2=np.copy(y)
    y2=np.abs(y2)
    y2=np.log(y2)
    plt.imsave(f"{chanelName}res10.jpg",y2,cmap='gray')
    y=np.fft.ifftshift(y)
    y=np.fft.ifft2(y)
    y=np.real(y)
    sharped_chanel=y
    sharped_chanel= np.clip(y, 0, 255).astype('uint8')
    return sharped_chanel

def Highpass_filter(sigma,width,height):
    filter=np.zeros((width,height),dtype='float64')
    filter=d_u_v_array(width,height)
    filter=filter/(2*(sigma**2))
    filter=-filter
    filter=np.exp(filter) 
    return 1-filter

def d_u_v_array(width,height):
    array=np.zeros((height,width),dtype='float64')
    a=np.arange(-width//2,width//2)
    # a=a-width/2
    a = np.expand_dims(a, axis=0)
    a=a**2
    u2=array+a
    a=np.arange(-height//2,height//2)
    # a=a-height/2
    a = np.expand_dims(a, axis=1)
    a=a**2
    v2=array+a
    array=u2+v2
    # print(array[height//2,width//2])
    return array

def sharp_chanel_section_D(image_chanel):
    image_fft=np.fft.fft2(image_chanel)
    image_fft_shifted=np.fft.fftshift(image_fft)
    x=4*(np.pi**2)*d_u_v_array_section_D(image_chanel.shape[1],image_chanel.shape[0])
    x=x*image_fft_shifted
    y=np.fft.ifftshift(x)
    y=np.fft.ifft2(y)
    y=np.real(y)
    y=y/np.max(np.abs(np.copy(y)))
    # print(np.max(y))
    # print(np.min(y))
    return y

def d_u_v_array_section_D(width,height):
    array=np.zeros((height,width),dtype='float64')
    # a=np.arange(-width//2,width//2)
    a=np.arange(width)
    a=a-width//2
    a = np.expand_dims(a, axis=0)
    a=a**2
    u2=array+a
    # a=np.arange(-height//2,height//2)
    a=np.arange(height)
    a=a-height//2
    a = np.expand_dims(a, axis=1)
    a=a**2
    v2=array+a
    array=u2+v2
    # print(array[height//2,width//2])
    return array

def show(image,chanelName):
    image_fft=np.fft.fft2(image)
    image_fft_shifted=np.fft.fftshift(image_fft)
    x=4*(np.pi**2)*d_u_v_array_section_D(image.shape[1],image.shape[0])
    x=x*image_fft_shifted
    x2=np.copy(x)
    x2=np.abs(x2)
    x2=np.log(x2+0.00001)
    # plt.imshow(x2)
    # plt.show()
    plt.imsave(f"{chanelName}res12.jpg",x2)
    y=np.fft.ifftshift(x)
    y=np.fft.ifft2(y)
    y=np.real(y)
    plt.imsave(f"{chanelName}res13.jpg",y,cmap='gray')


def first():
    alpha=4
    size=5
    sigma=2
    image=cv.imread("flowers.blur.png").astype('float64')
    # image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    blurred_image=cv.GaussianBlur(image,(size,size),sigma)
    cv.imwrite("res02.jpg",blurred_image)
    unsharp_mask=image-blurred_image
    visual_unsharp_mask=unsharp_mask
    visual_unsharp_mask=visual_unsharp_mask.astype('float64')
    visual_unsharp_mask=normalize(visual_unsharp_mask)
    visual_unsharp_mask= np.clip(visual_unsharp_mask, 0, 255)
    visual_unsharp_mask=visual_unsharp_mask.astype('uint8')
    image=image+alpha*unsharp_mask
    image[image>255]=255
    image[image<0]=0
    image=image.astype('uint8')
    cv.imwrite('res04.jpg',image)
    cv.imwrite("res03.jpg",visual_unsharp_mask)
    kernel=show_gaussain_filter(size,sigma)
    emp=np.zeros((201,201))
    for i in range(size):
        for j in range(size):
            a=(201-size)//2
            emp[a+i][a+j]=kernel[i,j]

    # plt.imshow(kernel)
    # plt.show()
    plt.imsave("res01.jpg",emp,cmap='gray')



def second():
    sigma=0.7
    size=3
    k=4
    image=cv.imread("flowers.blur.png").astype('float64')
    log_filter=Log_filter(sigma,size)
    log_filter=log_filter-np.mean(log_filter)

    unsharp_mask=np.zeros_like(image)
    unsharp_mask[:,:,0]=signal.correlate(image[:,:,0],log_filter,mode='same')
    unsharp_mask[:,:,1]=signal.correlate(image[:,:,1],log_filter,mode='same')
    unsharp_mask[:,:,2]=signal.correlate(image[:,:,2],log_filter,mode='same')
    visual_unsharp_mask=unsharp_mask
    visual_unsharp_mask=visual_unsharp_mask.astype('float64')
    visual_unsharp_mask=normalize(visual_unsharp_mask)
    visual_unsharp_mask= np.clip(visual_unsharp_mask, 0, 255)
    visual_unsharp_mask=visual_unsharp_mask.astype('uint8')
    cv.imwrite('res06.jpg',visual_unsharp_mask)
    sharped_image=image-k*unsharp_mask
    sharped_image[sharped_image>255]=255
    sharped_image[sharped_image<0]=0
    sharped_image=sharped_image.astype('uint8')
    cv.imwrite('res07.jpg',sharped_image)
    kernel=Log_filter(sigma,size)
    kernel=kernel-np.mean(log_filter)
    emp=np.zeros((201,201))
    for i in range(size):
        for j in range(size):
            a=(201-size)//2
            emp[a+i][a+j]=kernel[i,j]


    # plt.imshow(kernel)
    # plt.show()
    plt.imsave("res05.jpg",emp,cmap='gray')


def third():
    D0=60
    k=2
    image=cv.imread("flowers.blur.png")
    show_magnitude(image[:,:,0],'blue')
    show_magnitude(image[:,:,1],'green')
    show_magnitude(image[:,:,2],'red')



    hp2=Highpass_filter(D0,image.shape[1],image.shape[0])
    plt.imsave("res09.jpg",hp2,cmap='gray')

    image=rgb_sharping(image,60,k)
    cv.imwrite("res11.jpg",image)


def fourth():
    k=2
    image=cv.imread('flowers.blur.png')
    image=image.astype('float64')
    show(image[:,:,0],'blue')
    show(image[:,:,1],'green')
    show(image[:,:,2],'red')
    image=image/255
    sharped_image=np.zeros_like(image,dtype='float64')

    sharped_image[:,:,0]=image[:,:,0]+k*sharp_chanel_section_D(image[:,:,0])
    sharped_image[:,:,1]=image[:,:,1]+k*sharp_chanel_section_D(image[:,:,1])
    sharped_image[:,:,2]=image[:,:,2]+k*sharp_chanel_section_D(image[:,:,2])
    sharped_image=sharped_image*255
    sharped_image= np.clip(sharped_image, 0, 255).astype('uint8')
    cv.imwrite("res14.jpg.jpg",sharped_image)

first()
second()
third()
fourth()