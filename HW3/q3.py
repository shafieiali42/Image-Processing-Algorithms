import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt


def combineVerticalOverlaps(overlap1,new_overlap):
    mask=verticalMinimumCostPath(np.copy(overlap1),np.copy(new_overlap),1,0)
    full_one_matrix=np.ones_like(overlap1)
    result=mask*overlap1
    result=result+(full_one_matrix-mask)*new_overlap
    return result

def combineHorizantalOverlaps(overlap1,new_overlap):
    mask=horizantalMinimumCostPath(np.copy(overlap1),np.copy(new_overlap),1,0)
    full_one_matrix=np.ones_like(overlap1)
    result=mask*overlap1
    result=result+(full_one_matrix-mask)*new_overlap
    return result

def horizantalMinimumCostPath(overlap1,new_overlap,up,down):
    diff=np.zeros_like(overlap1,dtype='float64')
    diff=(overlap1-new_overlap)**2
    diff=np.sum(diff,axis=2)
    move_matrix=np.zeros_like(diff,dtype='int32')
    cost_matrix=np.zeros_like(diff,dtype='float64')
 
    for j in range(diff.shape[1]):
        for i in range(diff.shape[0]):
            if j==0:
                cost_matrix[i,j]=diff[i,j]
                move_matrix[i,j]=0
            else:
                cost=[]
                move=[]
                if i>0:
                    cost.append(int(diff[i,j])+int(cost_matrix[i-1,j-1]))
                    move.append(-1) #up
                cost.append(int(diff[i,j])+int(cost_matrix[i,j-1]))
                move.append(0) #same row
                if i<(diff.shape[0]-1):
                    cost.append(int(diff[i,j])+int(cost_matrix[i+1,j-1]))
                    move.append(1) #down
                min_cost=min(cost)
                min_move=move[cost.index(min_cost)]
                cost_matrix[i,j]=min_cost
                move_matrix[i,j]=min_move

        mask=np.ones((diff.shape[0],diff.shape[1],3))
        mask=mask*down
        min_index=np.argmin(cost_matrix[:,diff.shape[1]-1])
        mask[:min_index+1,diff.shape[1]-1]=up
        for j in range(diff.shape[1]-2,-1,-1):
            min_index=min_index+move_matrix[min_index,j+1]
            mask[:min_index+1,j,:]=up

        return mask



def verticalMinimumCostPath(overlap1,new_overlap,left,right):
    diff=np.zeros_like(overlap1,dtype='float64')
    diff=(overlap1-new_overlap)**2
    diff=np.sum(diff,axis=2)
    move_matrix=np.zeros_like(diff,dtype='int32')
    cost_matrix=np.zeros_like(diff,dtype='float64')
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if i==0:
                cost_matrix[i,j]=diff[i,j]
                move_matrix[i,j]=0
            else:
                cost=[]
                move=[]
                if j<(diff.shape[1]-1):
                    cost.append(int(diff[i,j])+int(cost_matrix[i-1,j+1]))
                    move.append(1) #right
                cost.append(int(diff[i,j])+int(cost_matrix[i-1,j]))
                move.append(0) #same col
                if j>0:
                    cost.append(int(diff[i,j])+int(cost_matrix[i-1,j-1]))
                    move.append(-1) #left
                
                min_cost=min(cost)
                min_move=move[cost.index(min_cost)]
                cost_matrix[i,j]=min_cost
                move_matrix[i,j]=min_move

    
    mask=np.ones((diff.shape[0],diff.shape[1],3))
    mask=mask*right
    min_index=np.argmin(cost_matrix[diff.shape[0]-1,:])
    mask[diff.shape[0]-1,:min_index+1,:]=left
    for i in range(diff.shape[0]-2,-1,-1):
        min_index=min_index+move_matrix[i+1,min_index]
        mask[i,:min_index+1,:]=left

    return mask

def matchOverlap(image,template,mask):
    match=cv.matchTemplate(image,template,method=cv.TM_CCOEFF_NORMED,mask=mask)

    x_index=[]
    y_index=[]
    match_rate=[]
    for i in range(10):
        min_value, max_value, min_loc, max_loc = cv.minMaxLoc(match)
        x = max_loc[0]
        x_index.append(x)
        match_rate.append(max_value)
        y = max_loc[1]
        y_index.append(y)
        match[y, x] = 0
    randomIndex=random.randint(0,len(x_index)-1)
    return y_index[randomIndex],x_index[randomIndex],match_rate[randomIndex]

def complete_rest_of_image(image,texture1,texture2,i,j,patch_size,overlap_size):
    vertical_overlap=np.copy(image[i:i+patch_size,j:j+overlap_size,:])
    horizantal_overlap=np.copy(image[i:i+overlap_size,j:j+patch_size,:])
    mask=np.zeros((patch_size,patch_size),dtype='uint8')
    mask[:,:overlap_size]=1
    mask[:overlap_size,:]=1
   
    
    x1,y1,rate1=matchOverlap(np.copy(texture1),np.copy(image[i:i+patch_size,j:j+patch_size,:]),mask)
    x2,y2,rate2=matchOverlap(np.copy(texture2),np.copy(image[i:i+patch_size,j:j+patch_size,:]),mask)
    if rate1>rate2:
        texture=texture1
        x=x1
        y=y1
    else:
        texture=texture2
        x=x2
        y=y2
    new_vertical_overlap=np.copy(texture[x:x+patch_size,y:y+overlap_size,:])
    new_horizantal_overlap=np.copy(texture[x:x+overlap_size,y:y+patch_size,:])
    vertical_mask=verticalMinimumCostPath(vertical_overlap,new_vertical_overlap,-1,1)
    horizantal_mask=horizantalMinimumCostPath(horizantal_overlap,new_horizantal_overlap,-1,1)
    g=np.ones((patch_size,patch_size,3))
    g[:overlap_size,:,:]=horizantal_mask
    g[:,:overlap_size,:]=g[:,:overlap_size,:]+vertical_mask
    new_mask = np.where(g<=0, 1, 0)
    full_one=np.ones((patch_size,patch_size,3))
    current_patch=np.copy(texture[x:x+patch_size,y:y+patch_size,:])
    before=np.copy(image[i:i+patch_size,j:j+patch_size,:])
    current_patch=before*new_mask+(full_one-new_mask)*current_patch   
    return current_patch 


def image_completion(image,top_left,bottom_right,patch_size,overlap_size,texture1,texture2):
    cropped_image=image[top_left[0]-patch_size:bottom_right[0],top_left[1]-patch_size:bottom_right[1]]
    height=cropped_image.shape[0]
    width=cropped_image.shape[1]
    for i in range(patch_size-overlap_size,height-patch_size+1,patch_size-overlap_size):
        for j in range(patch_size-overlap_size,width-patch_size+1,patch_size-overlap_size):
            current_patch=np.zeros((patch_size,patch_size,3),dtype='uint8')
            current_patch=complete_rest_of_image(cropped_image,texture1,texture2,i,j,patch_size,overlap_size)
            cropped_image[i:i+patch_size,j:j+patch_size,:]=current_patch
    
    return cropped_image


def reBuildRight(image,top_left,bottom_right,patch_size,overlap_size,texture1,texture2):
    cropped_image=image[top_left[0]-overlap_size:bottom_right[0],top_left[1]:bottom_right[1]+overlap_size,:]
    height=cropped_image.shape[0]
    width=cropped_image.shape[1]
    j=width-patch_size
    for i in range(0,height-patch_size+1,patch_size):
        before=np.copy(cropped_image[i:i+patch_size,j:j+patch_size,:])
        mask=np.ones((patch_size,patch_size),dtype='uint8')
        x1,y1,rate1=matchOverlap(np.copy(texture1),np.copy(cropped_image[i:i+patch_size,j:j+patch_size,:]),mask)
        x2,y2,rate2=matchOverlap(np.copy(texture2),np.copy(cropped_image[i:i+patch_size,j:j+patch_size,:]),mask)
        if rate1>rate2:
            texture=texture1
            x=x1
            y=y1
        else:
            texture=texture2
            x=x2
            y=y2

        current_patch=np.copy(texture[x:x+patch_size,y:y+patch_size,:])
        vertical_mask=verticalMinimumCostPath(before,current_patch,-1,1)
        horizantal_mask=horizantalMinimumCostPath(before,current_patch,-1,1)
        new_mask=vertical_mask+horizantal_mask
        new_mask=np.where(new_mask<=0,0,1)
        full_one=np.ones((patch_size,patch_size,3))
        current_patch=before*new_mask+(full_one-new_mask)*current_patch 
        vertical_overlap=np.copy(current_patch[:,:overlap_size,:])
        before_vertical_overlap=np.copy(cropped_image[i:i+patch_size,j-overlap_size:j,:])
        mixed_vertical_overlap=combineVerticalOverlaps(before_vertical_overlap,vertical_overlap)
        current_patch[:,:overlap_size,:]=mixed_vertical_overlap
        if i>0:
            horizantal_overlp=np.copy(current_patch[:overlap_size,:,:])
            before_horizantal_overlap=np.copy(cropped_image[i-overlap_size:i,j:j+patch_size,:])
            mixed_horizantal_overlap=combineHorizantalOverlaps(before_horizantal_overlap,horizantal_overlp)
            current_patch[:overlap_size,:,:]=mixed_horizantal_overlap

        cropped_image[i:i+patch_size,j:j+patch_size,:]=current_patch
    return cropped_image

def reBuildDown(image,top_left,bottom_right,patch_size,overlap_size,texture1,texture2):
    cropped_image=image[top_left[0]:bottom_right[0]+overlap_size,top_left[1]-overlap_size:bottom_right[1],:]
    height=cropped_image.shape[0]
    width=cropped_image.shape[1]
    i=height-patch_size
    for j in range(0,width-patch_size+1,patch_size):
        before=np.copy(cropped_image[i:i+patch_size,j:j+patch_size])
        mask=np.ones((patch_size,patch_size),dtype='uint8')
        x1,y1,rate1=matchOverlap(np.copy(texture1),np.copy(cropped_image[i:i+patch_size,j:j+patch_size,:]),mask)
        x2,y2,rate2=matchOverlap(np.copy(texture2),np.copy(cropped_image[i:i+patch_size,j:j+patch_size,:]),mask)
        if rate1>rate2:
            texture=texture1
            x=x1
            y=y1
        else:
            texture=texture2
            x=x2
            y=y2
        current_patch=np.copy(texture[x:x+patch_size,y:y+patch_size,:])
        vertical_mask=verticalMinimumCostPath(before,current_patch,-1,1)
        horizantal_mask=horizantalMinimumCostPath(before,current_patch,-1,1)
        new_mask=vertical_mask+horizantal_mask
        new_mask=np.where(new_mask<=0,0,1)
        full_one=np.ones((patch_size,patch_size,3))
        current_patch=before*new_mask+(full_one-new_mask)*current_patch
        if j>0:
            vertical_overlap=np.copy(current_patch[:,:overlap_size,:])
            before_vertical_overlap=np.copy(cropped_image[i:i+patch_size,j-overlap_size:j,:])
            mixed_vertical_overlap=combineVerticalOverlaps(before_vertical_overlap,vertical_overlap)
            current_patch[:,:overlap_size,:]=mixed_vertical_overlap
        
        horizantal_overlp=np.copy(current_patch[:overlap_size,:,:])
        before_horizantal_overlap=np.copy(cropped_image[i-overlap_size:i,j:j+patch_size,:])
        mixed_horizantal_overlap=combineHorizantalOverlaps(before_horizantal_overlap,horizantal_overlp)
        current_patch[:overlap_size,:,:]=mixed_horizantal_overlap
        cropped_image[i:i+patch_size,j:j+patch_size]=current_patch
    return cropped_image











image=cv.imread('im03.jpg')
image[60:210,320:570]=0
image[740:940,800:1000]=0
image[585:720,1110:1245]=0
patch_size=60
overlap_size=10
texture1=image[340:540,755:958]
texture2=image[193:327,561:708]

top_left=(60,320)
bottom_right=(210,570)
fill=image_completion(image,top_left,bottom_right,patch_size,overlap_size,texture2,texture2)
image[top_left[0]-patch_size:bottom_right[0],top_left[1]-patch_size:bottom_right[1]]=fill

fill=reBuildRight(image,top_left,bottom_right,patch_size,overlap_size,texture2,texture2)
image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]=fill[overlap_size:,:fill.shape[1]-overlap_size]
fill=reBuildDown(image,top_left,bottom_right,patch_size,overlap_size,texture2,texture2)
image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]=fill[:fill.shape[0]-overlap_size,overlap_size:]


patch_size=60
overlap_size=10
top_left=(740,800)
bottom_right=(940,1000)
fill=image_completion(image,top_left,bottom_right,patch_size,overlap_size,texture1,texture1)
image[top_left[0]-patch_size:bottom_right[0],top_left[1]-patch_size:bottom_right[1]]=fill
fill=reBuildRight(image,top_left,bottom_right,patch_size,overlap_size,texture1,texture1)
image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]=fill[overlap_size:,:fill.shape[1]-overlap_size]
fill=reBuildDown(image,top_left,bottom_right,patch_size,overlap_size,texture1,texture1)
image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]=fill[:fill.shape[0]-overlap_size,overlap_size:]


patch_size=50
overlap_size=5
top_left=(585,1110)
bottom_right=(720,1245)
fill=image_completion(image,top_left,bottom_right,patch_size,overlap_size,texture1,texture1)
image[top_left[0]-patch_size:bottom_right[0],top_left[1]-patch_size:bottom_right[1]]=fill
fill=reBuildRight(image,top_left,bottom_right,patch_size,overlap_size,texture1,texture1)
image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]=fill[overlap_size:,:fill.shape[1]-overlap_size]
fill=reBuildDown(image,top_left,bottom_right,patch_size,overlap_size,texture1,texture1)
image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]=fill[:fill.shape[0]-overlap_size,overlap_size:]

cv.imwrite("res15.jpg",image)


image=cv.imread('im04.jpg')
image[670:1170,730:980]=0
patch_size=60
overlap_size=10
texture1=image[458:687,534:685]
texture2=image[1027:1316,1025:1239]
top_left=(670,730)
bottom_right=(1170,980)
fill=image_completion(image,top_left,bottom_right,patch_size,overlap_size,texture2,texture2)
image[top_left[0]-patch_size:bottom_right[0],top_left[1]-patch_size:bottom_right[1]]=fill
fill=reBuildRight(image,top_left,bottom_right,patch_size,overlap_size,texture2,texture2)
image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]=fill[overlap_size:,:fill.shape[1]-overlap_size]
fill=reBuildDown(image,top_left,bottom_right,patch_size,overlap_size,texture2,texture2)
image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]=fill[:fill.shape[0]-overlap_size,overlap_size:]
cv.imwrite("res16.jpg",image)


