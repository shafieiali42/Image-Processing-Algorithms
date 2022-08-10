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

def selectRandomPatch(texture,size):
    h1=random.randint(0,texture.shape[1]-size)
    # h2=random.randint(size-1,texture.shape[1]-1)
    w1=random.randint(0,texture.shape[0]-size)
    # w2=random.randint(size-1,texture.shape[0]-1)
    sample=np.copy(texture[w1:w1+size,h1:h1+size,:])
    return sample

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
    # print(mask.shape)
    return mask
    
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




def matchOverlap(image,template,mask):
    match=cv.matchTemplate(image,template,method=cv.TM_CCOEFF_NORMED,mask=mask)
    x_index=[]
    y_index=[]
    for i in range(10):
        min_value, max_value, min_loc, max_loc = cv.minMaxLoc(match)
        x = max_loc[0]
        x_index.append(x)
        y = max_loc[1]
        y_index.append(y)
        match[y, x] = 0
    randomIndex=random.randint(0,len(x_index)-1)
    return y_index[randomIndex],x_index[randomIndex]


def complete_first_row(image,texture,i,j,patch_size,overlap_size):
    overlap=np.copy(image[i:i+patch_size,j:j+overlap_size,:])
    mask=np.zeros((patch_size,patch_size),dtype='uint8')
    mask[:,:overlap_size]=1
    x,y=matchOverlap(np.copy(texture),np.copy(image[i:i+patch_size,j:j+patch_size]),mask)
    current_patch=np.copy(texture[x:x+patch_size,y:y+patch_size,:])
    new_overlap=np.copy(texture[x:x+patch_size,y:y+overlap_size,:])
    mixed_overlap=combineVerticalOverlaps(overlap,new_overlap)
    current_patch[:,:overlap_size,:]=mixed_overlap    
    return current_patch
    
def complete_first_column(image,texture,i,j,patch_size,overlap_size):
    overlap=np.copy(image[i:i+overlap_size,j:j+patch_size,:])
    mask=np.zeros((patch_size,patch_size),dtype='uint8')
    mask[:overlap_size,:]=1
    x,y=matchOverlap(np.copy(texture),np.copy(image[i:i+patch_size,j:j+patch_size]),mask)
    current_patch=np.copy(texture[x:x+patch_size,y:y+patch_size,:])
    new_overlap=np.copy(texture[x:x+overlap_size,y:y+patch_size,:])
    mixed_overlap=combineHorizantalOverlaps(overlap,new_overlap)
    current_patch[:overlap_size,:,:]=mixed_overlap
    return current_patch


def complete_rest_of_image(image,texture,i,j,patch_size,overlap_size):
    vertical_overlap=np.copy(image[i:i+patch_size,j:j+overlap_size,:])
    horizantal_overlap=np.copy(image[i:i+overlap_size,j:j+patch_size,:])
    mask=np.zeros((patch_size,patch_size),dtype='uint8')
    mask[:,:overlap_size]=1
    mask[:overlap_size,:]=1
    x,y=matchOverlap(np.copy(texture),np.copy(image[i:i+patch_size,j:j+patch_size,:]),mask)
    new_vertical_overlap=np.copy(texture[x:x+patch_size,y:y+overlap_size,:])
    new_horizantal_overlap=np.copy(texture[x:x+overlap_size,y:y+patch_size,:])
    vertical_mask=verticalMinimumCostPath(vertical_overlap,new_vertical_overlap,-1,1)
    horizantal_mask=horizantalMinimumCostPath(horizantal_overlap,new_horizantal_overlap,-1,1)
    new_temp_mask=np.ones((patch_size,patch_size,3))
    new_temp_mask[:overlap_size,:,:]=horizantal_mask
    new_temp_mask[:,:overlap_size,:]=new_temp_mask[:,:overlap_size,:]+vertical_mask
    new_mask = np.where(new_temp_mask<=0, 1, 0)
    full_one=np.ones((patch_size,patch_size,3))
    current_patch=np.copy(texture[x:x+patch_size,y:y+patch_size,:])
    before=np.copy(image[i:i+patch_size,j:j+patch_size,:])
    current_patch=before*new_mask+(full_one-new_mask)*current_patch   
    return current_patch 


def imageQuilting(texture,patch_size,overlap_size,height,width):
    image=np.zeros((width,height,3),dtype='uint8')
    for i in range(0,height-patch_size+1,patch_size-overlap_size):
        for j in range(0,width-patch_size+1,patch_size-overlap_size):
            current_patch=np.zeros((patch_size,patch_size,3),dtype='uint8')
            if i==0:
                if j==0:
                    current_patch=selectRandomPatch(texture,patch_size)
                else:
                    current_patch=complete_first_row(image,texture,i,j,patch_size,overlap_size)
            elif j==0:
                current_patch=complete_first_column(image,texture,i,j,patch_size,overlap_size)
            else:
                current_patch=complete_rest_of_image(image,texture,i,j,patch_size,overlap_size)
            
            image[i:i+patch_size,j:j+patch_size,:]=current_patch
    return image




texture1=cv.imread('texture06.jpg')
# texture2=cv.imread('texture11.jpeg')
# texture3=cv.imread('myTexture3.jpg')
# texture4=cv.imread('myTexture2.jpg')
image1=imageQuilting(texture1,250,25,2500,2500)
# image2=imageQuilting(texture2,250,25,2500,2500)
# image3=imageQuilting(texture3,250,25,2500,2500)
# image4=imageQuilting(texture4,250,25,2500,2500)
cv.imwrite('res11.jpg',image1)
# cv.imwrite('res12.jpg',image2)
# cv.imwrite('res13.jpg',image3)
# cv.imwrite('res14.jpg',image4)
