import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def show_magnitude(image,chanelName,number,name):
    image2=np.copy(image)
    image2=image2.astype('float64')
    image_fft=np.fft.fft2(image2)
    fft_shifted=np.fft.fftshift(image_fft)
    amplitude_image=np.abs(fft_shifted)
    log_image=np.log(amplitude_image)
    plt.imsave(f"{chanelName}_res{number}-dft-{name}.jpg",log_image,cmap='gray')


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

def lowpass_filter(sigma,width,height):
    filter=np.zeros((width,height),dtype='float64')
    filter=d_u_v_array(width,height)
    filter=filter/(2*(sigma**2))
    filter=-filter
    filter=np.exp(filter)
    # lowpass= filter-np.mean(filter)   
    return filter

def highpass_filter(sigma,width,height):
    highpass=1-lowpass_filter(sigma,width,height)
    return highpass


far_image=cv.imread("res20-far.jpg")
near_image=cv.imread("res19-near.jpg")
cv.imwrite("res21-near.jpg",near_image)
cv.imwrite("res22-far.jpg",far_image)
show_magnitude(near_image[:,:,0],"blue",23,"near")
show_magnitude(near_image[:,:,1],"green",23,"near")
show_magnitude(near_image[:,:,2],"red",23,"near")


show_magnitude(far_image[:,:,0],"blue",24,"far")
show_magnitude(far_image[:,:,1],"green",24,"far")
show_magnitude(far_image[:,:,2],"red",24,"far")

hp=highpass_filter(19,near_image.shape[1],near_image.shape[0])
plt.imsave("res25-highpass-10.jpg",hp,cmap='gray')

lp=lowpass_filter(10,near_image.shape[1],near_image.shape[0])
plt.imsave("res26-lowpass-19.jpg",lp,cmap='gray')


def align_picture(dst,far_coor,near_coor,width,height):
    t_mat = cv.estimateAffine2D(far_coor, near_coor)[0]
    aligned_picture = cv.warpAffine(dst, t_mat,(width,height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)
    return aligned_picture


def calculate_fft_image(image):
    image_fft=np.fft.fft2(image)
    shifted_fft=np.fft.fftshift(image_fft)
    return shifted_fft


def filtering(image_fft,filter):
    result=image_fft*filter
    return result



def combine_two_chanel(image1,image2):
    im1=image1/2
    im2=image2/2
    total=im1+im2
    return total


def make_hybrid_image(near,far,highpass_cutoff,lowpass_cutoff):
    red_near=near[:,:,2]
    green_near=near[:,:,1]
    blue_near=near[:,:,0]
    red_far=far[:,:,2]
    green_far=far[:,:,1]
    blue_far=far[:,:,0]
    red=one_chanel(red_near,red_far,highpass_cutoff[2],lowpass_cutoff[2],'red')
    green=one_chanel(green_near,green_far,highpass_cutoff[1],lowpass_cutoff[1],'green')
    blue=one_chanel(blue_near,blue_far,highpass_cutoff[0],lowpass_cutoff[0],'blue')
    result=np.zeros_like(near)
    result[:,:,2]=red
    result[:,:,1]=green
    result[:,:,0]=blue
    return result



def one_chanel(near,far,highpass_cutoff,lowpass_cutoff,chanelName):
    highpassed_near=filtering(calculate_fft_image(near),
    highpass_filter(highpass_cutoff,near.shape[1],near.shape[0]))
    visual_highpass_near=np.copy(highpassed_near)
    visual_highpass_near=np.abs(visual_highpass_near)
    visual_highpass_near=np.log(visual_highpass_near+1)
    plt.imsave(f"{chanelName}_res27-highpassed.jpg",visual_highpass_near,cmap='gray')


    lowpass_far=filtering(calculate_fft_image(far),
    lowpass_filter(lowpass_cutoff,far.shape[1],far.shape[0]))
    visual_lowpass_far=np.copy(lowpass_far)
    visual_lowpass_far=np.abs(visual_lowpass_far)
    visual_lowpass_far=np.log(visual_lowpass_far+1)
    plt.imsave(f"{chanelName}res28-lowpassed.jpg",visual_lowpass_far,cmap='gray')


    result=combine_two_chanel(highpassed_near,lowpass_far)
    visual_result=np.copy(result)
    visual_result=np.abs(visual_result)
    visual_result=np.log(visual_result+1)
    plt.imsave(f"{chanelName}res29-hybrid.jpg",visual_result,cmap='gray')

    result=np.fft.ifftshift(result)
    result=np.fft.ifft2(result)
    result=np.real(result)
    result= cv.normalize(result, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
    result[result>255]=255
    result[result<0]=0
    # print(np.max(result))
    # print(np.min(result))
    return result


# def find_cut_off(image):
#     image=image.astype('float64')
#     image_fft=np.fft.fft2(image)
#     shifted_fft=np.fft.fftshift(image_fft)
#     amplitude=np.abs(shifted_fft)
#     amplitude2=amplitude/2

#     total_sum=np.sum(amplitude2)
#     uv_array=d_u_v_array(image.shape[1],image.shape[0])
#     uv_array=np.sqrt(uv_array)
#     diff=np.zeros(100)
#     sums=np.zeros(100)
#     for i in range(100):
#         uv_array2=np.copy(uv_array)
#         uv_array2=uv_array2.astype('float64')
#         uv_array2[uv_array2>i]=-1
#         uv_array2[uv_array2!=-1]=1
#         uv_array2[uv_array2==-1]=0
#         res=uv_array2*amplitude
#         sum=np.sum(res)
#         sums[i]=sum
#         diff[i]=np.abs(sum-total_sum)

#     print("Res:")
#     min_index=np.argmin(diff)
#     print(sums[min_index])
#     print(diff[min_index])
#     print("*******")
#     print(min_index)
#     return min_index



aligned_far=far_image
near_image=near_image.astype('float64')
aligned_far=aligned_far.astype('float64')


a=10
high_cutoff=[a,a,a]
b=19
low_cutoff=[b,b,b]
result=make_hybrid_image(np.copy(near_image),np.copy(aligned_far),high_cutoff,low_cutoff)
result=result.astype('uint8')
cv.imwrite("res30-hybrid-near.jpg",result)
small_result=cv.resize(result, (result.shape[0]//5,result.shape[1]//5), interpolation = cv.INTER_AREA)
cv.imwrite("res31-hybrid-far.jpg",small_result)





