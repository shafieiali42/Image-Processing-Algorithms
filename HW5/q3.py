import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time


def make_laplacian_stack( image, size_of_stack, kernel_size, sigma,a):
    stack=[]

    for i in range(size_of_stack - 1):
        image2 = cv.GaussianBlur(image, (kernel_size, kernel_size),sigma)
        stack.append(image - image2)
        # cv.imwrite(f"laplacian{a}{i}.jpg",(stack[i].copy().astype('uint8')))

        image = image2
        sigma=sigma*2
    # image=cv.GaussianBlur(image, (kernel_size, kernel_size),sigma)
    stack.append(image)
    # cv.imwrite(f"laplacian{a}{size_of_stack-1}.jpg", stack[size_of_stack-1].copy().astype('uint8'))

    return stack


def make_gaussian_stack(image, size_of_stack, kernel_size, sigma):
    counter = 0
    stack=[]
    for i in range(size_of_stack):
        image2 = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        stack.append(image2)
        sigma=sigma*2
    return stack


def blend(source_image,target_image,source_mask,size_of_stack,kernel_size,sigma):
    source_laplacian_stack=make_laplacian_stack(source_image,size_of_stack,kernel_size,sigma,'src')
    target_laplacian_stack=make_laplacian_stack(target_image,size_of_stack,kernel_size,sigma,'tar')
    mask_gaussian_stack=make_gaussian_stack(source_mask,size_of_stack,kernel_size,sigma)
    composite_stack=[]
    for i in range(size_of_stack):
        # cv.imwrite(f"maskGaussian{i}.jpg", (mask_gaussian_stack[i].copy().astype('uint8')))
        new_mask=mask_gaussian_stack[i]/np.max(mask_gaussian_stack[i])
        composite=new_mask*source_laplacian_stack[i]+(1.0-new_mask)*target_laplacian_stack[i]
        composite_stack.append(composite)
        # cv.imwrite(f"composite{i}.jpg", (composite_stack[i].copy()))

    result_image=np.zeros_like(source_image,dtype='float64')

    for i in range(size_of_stack):
        result_image=result_image+composite_stack[i]


    return result_image


def q3(target,source_name,mask_name,target_point,size_of_stack,kernel_size,sigma,result):
    source_image=cv.imread(source_name)
    source_image=source_image.astype('float64')
    mask=cv.imread(mask_name)
    target_image=target.astype('float64')
    mask = mask.astype('float64')
    mask_white_index = np.where(mask > 0)
    target_top_left = target_point
    target_rect = target_image[target_top_left[0]:target_top_left[0] + source_image.shape[0],
                  target_top_left[1]:target_top_left[1] + source_image.shape[1]]

    result=result.astype('float64')
    result_rect = blend(source_image, target_rect, mask, size_of_stack, kernel_size, sigma)
    result[target_top_left[0]:target_top_left[0] + source_image.shape[0],
                target_top_left[1]:target_top_left[1] + source_image.shape[1]]=result_rect


    return result


start_time=time.time()
target=cv.imread("res09.jpg")
# result=target.copy()
result=np.zeros_like(target)
result=np.copy(target)
result=result.astype('float64')
result=q3(target,"res08.jpg","city_mask2.png",(306,0),5,45,2,result)
# print(np.max(result))
# print(np.min(result))
result=np.clip(result,0,255)
result=result.astype('uint8')
cv.imwrite("res10.jpg",result)
print(time.time()-start_time)
