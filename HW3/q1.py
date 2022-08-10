import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math





def createHoughSpace(edge_image,diag,diags,thetas,height,width):
    hough_space=np.zeros((len(diags),len(thetas)))
    cos_vector=np.cos(np.deg2rad(thetas))
    sin_vector=np.sin(np.deg2rad(thetas))
    a=np.where(edge_image==255)
    edge_x_indexes=a[0]  
    edge_y_indexes=a[1]
    edge_x_indexes=edge_x_indexes-height/2
    edge_y_indexes=edge_y_indexes-width/2
    points=np.zeros((edge_x_indexes.shape[0],2))
    points[:,0]=edge_x_indexes
    points[:,1]=edge_y_indexes
    sin_cos=np.zeros((2,180))
    sin_cos[0,:]=sin_vector
    sin_cos[1,:]=cos_vector
    R=np.matmul(points,sin_cos)

    for i in range(180):
        r=R[:,i]
        r=r+diag/2
        r=np.floor(r)
        r=r.astype('int32')
        values=np.ones_like(r)
        hough_space[:,i]+=np.bincount(r,values,len(diags))
   
    hough_space=np.floor(hough_space)
    hough_space=hough_space.astype('int32')
    return hough_space




def detecLines(hough_space,threshold,max_number):
    lines=np.zeros((max_number,2))
    number_of_lines=0
    for i in range(max_number):
        a=np.where(hough_space>threshold)
        
        line_r_index=a[0]
        line_theta_index=a[1]
        if line_r_index.size<=0:
            break
        
        b=np.where(hough_space==np.max(hough_space))
        line_r_index=b[0]
        line_theta_index=b[1]
        lines[i]=np.array([line_r_index[0],line_theta_index[0]])
        number_of_lines=number_of_lines+1
        space_r=15
        space_theta=20
        hough_space[max(0,line_r_index[0]-space_r):min(line_r_index[0]+space_r,hough_space.shape[0]-1),max(0,line_theta_index[0]-space_theta):min(line_theta_index[0]+space_theta,hough_space.shape[1]-1)]=0
    
    # print(f'Number of lines: {number_of_lines}')
    new_lines=np.zeros((number_of_lines,2))
    new_lines[:,:]=lines[:number_of_lines,:]
    return new_lines

    
def group_lines(lines):
    blueLines=np.zeros((lines.shape[0],2))
    orangeLines=np.zeros((lines.shape[0],2))
    number_of_blues=0
    number_of_oranges=0
    for i in range(lines.shape[0]):
        if lines[i,1]>=90:
            blueLines[number_of_blues]=np.array([lines[i,0],lines[i,1]])
            number_of_blues=number_of_blues+1
        else:
            orangeLines[number_of_oranges]=np.array([lines[i,0],lines[i,1]])
            number_of_oranges=number_of_oranges+1
    

    new_blues=blueLines[:number_of_blues,:]
    new_oranges=orangeLines[:number_of_oranges,:]
    return new_blues,new_oranges

def get_two_point_of_line(line,diags,thetas,height,width,len,shift=True):
    line=line.astype('int64')
    cos_value=np.cos(np.deg2rad(thetas[line[1]]))
    sin_value=np.sin(np.deg2rad(thetas[line[1]]))
    x = diags[line[0]]*cos_value
    y = diags[line[0]]*sin_value 
    x1 = x - len * sin_value
    y1 = y + len * cos_value
    x2 = x + len * sin_value
    y2 = y - len * cos_value
    if(shift):
        x1=int(x1+width/2)
        y1=int(y1+height/2)
        x2=int(x2+width/2)
        y2=int(y2+height/2)
    else:
        x1=int(x1)
        y1=int(y1)
        x2=int(x2)
        y2=int(y2)

    return (x1,y1),(x2,y2)

        
def draw_lines(lines,image,color):
    height=image.shape[0]
    width=image.shape[1]
    diag=np.sqrt(height**2+width**2)
    thetas=np.arange(0,180,1)
    diags=np.arange(-diag/2,diag/2,1)
    lines=lines.astype('int64')
    for i in range(lines.shape[0]):
        pt1,pt2=get_two_point_of_line(lines[i],diags,thetas,height,width,800,shift=True)
        cv.line(image,pt1,pt2,color,thickness=2)
    


def isBlack(mean,blackThreshold):
    return mean<=blackThreshold

def isWhite(mean,whiteThreshold):
    return mean>=whiteThreshold


def detect_good_lines(image,blueLines,orangeLines,diags,thetas,height,width,blackThreshold,whiteThreshold):

    good_blue_lines=np.zeros((blueLines.shape[0],2))
    good=np.zeros((blueLines.shape[0]+orangeLines.shape[0],2))
    number_of_good_blue_lines=0
    for i in range(blueLines.shape[0]):
        flag=False
        for j in range(orangeLines.shape[0]):
            if not flag:
                    pt1,pt2=get_two_point_of_line(blueLines[i],diags,thetas,height,width,800,shift=False) 
                    pt3,pt4=get_two_point_of_line(orangeLines[j],diags,thetas,height,width,800,shift=False) 
                    intersec=find_intersection_point(pt1,pt2,pt3,pt4)
                    a=int(intersec[0]+width/2)
                    b=int(intersec[1]+height/2)
                    intersec=(a,b)

                    # up_point=image[intersec[1],intersec[0]-10]
                    # down_point=image[intersec[1],intersec[0]+30]
                    # left_point=image[intersec[1]-10,intersec[0]]
                    # right_point=image[intersec[1]+10,intersec[0]]

                    up_point=image[intersec[1],intersec[0]-20]
                    down_point=image[intersec[1],intersec[0]+20]
                    left_point=image[intersec[1]-20,intersec[0]]
                    right_point=image[intersec[1]+20,intersec[0]]


                    # up_point=image[intersec[1]+10:intersec[1]+15,intersec[0]-15:intersec[0]-10]
                    # down_point=image[intersec[1]+10:intersec[1]+15,intersec[0]+10:intersec[0]+15]
                    # left_point=image[intersec[1]-15:intersec[1]-10,intersec[0]+10:intersec[0]+15]
                    # right_point=image[intersec[1]+10:intersec[1]+15,intersec[0]+10:intersec[0]+15]

                    black_cnt=0
                    white_cnt=0
                    upvalue=np.mean(up_point)
                    downvalue=np.mean(down_point)
                    rightvalue=np.mean(right_point)
                    leftvalue=np.mean(left_point)
                    if isBlack(upvalue,blackThreshold):
                        black_cnt=black_cnt+1
                    elif isWhite(upvalue,whiteThreshold):
                        white_cnt=white_cnt+1

                    if isBlack(downvalue,blackThreshold):
                        black_cnt=black_cnt+1
                    elif isWhite(downvalue,whiteThreshold):
                        white_cnt=white_cnt+1

                    if isBlack(rightvalue,blackThreshold):
                        black_cnt=black_cnt+1
                    elif isWhite(rightvalue,whiteThreshold):
                        white_cnt=white_cnt+1

                    if isBlack(leftvalue,blackThreshold):
                        black_cnt=black_cnt+1
                    elif isWhite(leftvalue,whiteThreshold):
                        white_cnt=white_cnt+1
                    
                   
                    if white_cnt==2:
                        flag=True
                    elif white_cnt==3:
                            flag=True
                    elif black_cnt==1:
                        flag=True
                    
                    if flag:
                        good_blue_lines[number_of_good_blue_lines]=blueLines[i]
                        number_of_good_blue_lines=number_of_good_blue_lines+1
                


    good_orangeLines=np.zeros((orangeLines.shape[0],2))
    number_of_good_orange_lines=0
    for j in range(orangeLines.shape[0]):
        flag=False
        for i in range(blueLines.shape[0]):
            if not flag:
                pt1,pt2=get_two_point_of_line(blueLines[i],diags,thetas,height,width,800,shift=False) 
                pt3,pt4=get_two_point_of_line(orangeLines[j],diags,thetas,height,width,800,shift=False) 
                intersec=find_intersection_point(pt1,pt2,pt3,pt4)
                a=int(intersec[0]+width/2)
                b=int(intersec[1]+height/2)
                intersec=(a,b)

                up_point=image[intersec[1],intersec[0]-20]
                down_point=image[intersec[1],intersec[0]+20]
                left_point=image[intersec[1]-20,intersec[0]]
                right_point=image[intersec[1]+20,intersec[0]]

                # up_point=image[intersec[1]+10:intersec[1]+15,intersec[0]-15:intersec[0]-10]
                # down_point=image[intersec[1]+10:intersec[1]+15,intersec[0]+10:intersec[0]+15]
                # left_point=image[intersec[1]-15:intersec[1]-10,intersec[0]+10:intersec[0]+15]
                # right_point=image[intersec[1]+10:intersec[1]+15,intersec[0]+10:intersec[0]+15]

                black_cnt=0
                white_cnt=0
                upvalue=np.mean(up_point)
                downvalue=np.mean(down_point)
                rightvalue=np.mean(right_point)
                leftvalue=np.mean(left_point)
                if isBlack(upvalue,blackThreshold):
                    black_cnt=black_cnt+1
                elif isWhite(upvalue,whiteThreshold):
                    white_cnt=white_cnt+1

                if isBlack(downvalue,blackThreshold):
                    black_cnt=black_cnt+1
                elif isWhite(downvalue,whiteThreshold):
                    white_cnt=white_cnt+1

                if isBlack(rightvalue,blackThreshold):
                    black_cnt=black_cnt+1
                elif isWhite(rightvalue,whiteThreshold):
                    white_cnt=white_cnt+1

                if isBlack(leftvalue,blackThreshold):
                    black_cnt=black_cnt+1
                elif isWhite(leftvalue,whiteThreshold):
                    white_cnt=white_cnt+1
                
                
                if white_cnt==2:
                    flag=True
                elif white_cnt==3:
                    flag=True
                elif black_cnt==1:
                        flag=True
                
                if flag:
                    good_orangeLines[number_of_good_orange_lines]=orangeLines[j]
                    number_of_good_orange_lines=number_of_good_orange_lines+1
        
  
    
    # print(f'Number of blue lines:{blueLines.shape[0]}')
    # print(f'Number of orange lines:{orangeLines.shape[0]}')
    # print(f'Number of good blue lines:{number_of_good_blue_lines}')
    # print(f'Number of good orange lines:{number_of_good_orange_lines}')
    new_good_blue_lines=np.zeros((number_of_good_blue_lines,2))
    new_good_blue_lines[:,:]=good_blue_lines[:number_of_good_blue_lines,:]

    new_good_orange_lines=np.zeros((number_of_good_orange_lines,2))
    new_good_orange_lines[:,:]=good_orangeLines[:number_of_good_orange_lines,:]

    return new_good_blue_lines,new_good_orange_lines

        


def find_intersection_point(pt1,pt2,pt3,pt4):
    denominator=(pt1[0]-pt2[0])*(pt3[1]-pt4[1])-(pt1[1]-pt2[1])*(pt3[0]-pt4[0])
    x=(pt1[0]*pt2[1]-pt1[1]*pt2[0])*(pt3[0]-pt4[0])-(pt3[0]*pt4[1]-pt4[0]*pt3[1])*(pt1[0]-pt2[0])
    x=x//denominator
    y=(pt1[0]*pt2[1]-pt1[1]*pt2[0])*(pt3[1]-pt4[1])-(pt3[0]*pt4[1]-pt4[0]*pt3[1])*(pt1[1]-pt2[1])
    y=y//denominator
    return (x,y)





    
def draw_Intersection(image,blueLines,orangeLines,diags,thetas,height,width):
    for i in range(blueLines.shape[0]):
        for j in range(orangeLines.shape[0]):
            pt1,pt2=get_two_point_of_line(blueLines[i],diags,thetas,height,width,800)
            pt3,pt4=get_two_point_of_line(orangeLines[j],diags,thetas,height,width,800)
            intersect=find_intersection_point(pt1,pt2,pt3,pt4)
            cv.circle(image, intersect, radius=10, color=(0, 0, 255), thickness=-1)




def question1(image,edgeName,houghSpaceName,lineName,goodLineImageName,intersectionImage):
    copy_image_good_line=np.copy(image)
    copy_image_line=np.copy(image)
    copy_image_intersection=np.copy(image)
    edge_image=cv.Canny(image,400,400)
    cv.imwrite(f'{edgeName}.jpg',edge_image)
    height=edge_image.shape[0]
    width=edge_image.shape[1]
    diag=np.sqrt(height**2+width**2)
    thetas=np.arange(0,180,1)
    diags=np.arange(-diag/2,diag/2,1)
    hough_space=createHoughSpace(edge_image,diag,diags,thetas,height,width)
    copy=np.copy(hough_space)
    copy=copy.astype('uint8')
    cv.imwrite(f'{houghSpaceName}.jpg',copy)
    lines=detecLines(hough_space,90,30)
    blue,orange=group_lines(lines)
    draw_lines(blue,copy_image_line,(255,255,0))
    draw_lines(orange,copy_image_line,(0,128,255))
    good_blue,good_orange=detect_good_lines(copy_image_good_line,blue,orange,diags,thetas,height,width,85,180)
    draw_lines(good_blue,copy_image_good_line,(255,215,0))
    draw_lines(good_orange,copy_image_good_line,(238,130,238))
    draw_Intersection(copy_image_intersection,good_blue,good_orange,diags,thetas,height,width)
    cv.imwrite(f'{intersectionImage}.jpg',copy_image_intersection)
    cv.imwrite(f'{goodLineImageName}.jpg',copy_image_good_line)
    cv.imwrite(f'{lineName}.jpg',copy_image_line)

image1=cv.imread('im01.jpg')
image2=cv.imread('im02.jpg')
# print(np.mean(image1[381,775]))
question1(image1,'res01','res03-hough-space','res05-lines','res07-chess','res09-corners')
question1(image2,'res02','res04-hough-space','res06-lines','res08-chess','res10-corners')


