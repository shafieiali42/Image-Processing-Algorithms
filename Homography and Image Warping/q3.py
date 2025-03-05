import cv2 as cv
import numpy as np
import time

a_coordinate = np.float32([
    [208, 662],
    [391, 601],
    [108, 379],
    [288, 319]
])

b_coordinate=np.array([
    [740,358],
    [708,155],
    [466,404],
    [428,208]
 ])

c_coordinate=np.array([
    [967,809],
    [1094,612],
    [670,619],
    [794,425]   
   ])


# def extract_book_opencv(books,book_coordinate,width,height,name):
#     book_dst_coordinate = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
#     matrix,status = cv.findHomography(book_coordinate, book_dst_coordinate)
#     extracted_book = cv.warpPerspective(books, matrix, (width, height))
#     cv.imwrite(f"{name}.jpg", extracted_book)


def extract_book(books,book_coordinate,width,height,name):
    width*=2
    height*=2
    books_red=books[:,:,2]
    books_blue=books[:,:,0]
    books_green=books[:,:,1]    
    book_dst_coordinate = np.float32([[0, 0], [0, width], [height,0], [height, width]])
    matrix,status = cv.findHomography(book_coordinate, book_dst_coordinate)
    # matrix=np.transpose(matrix)
    extracted_book_red = warp_image(books_red, matrix,width, height)
    extracted_book_blue = warp_image(books_blue, matrix,width, height)
    extracted_book_green = warp_image(books_green, matrix, width, height)
    extracted_book=np.zeros((height,width,3))
    extracted_book[:,:,0]=extracted_book_blue
    extracted_book[:,:,1]=extracted_book_green
    extracted_book[:,:,2]=extracted_book_red
    extracted_book = cv.GaussianBlur(extracted_book, (3, 3), 0)
    extracted_book = cv.resize(extracted_book, (0, 0), extracted_book, 0.5, 0.5, cv.INTER_LINEAR)
    cv.imwrite(f"{name}.jpg", extracted_book)


def bilinear_interpolation(image,x,y):
    a=x-int(x)
    b=int(y)-y
    c=np.array([[1-a,a]])
    d=np.array([
        [image[int(x),int(y)],image[int(x),int(y)+1]],
        [image[int(x)+1,int(y)],image[int(x)+1,int(y)+1]]
    ])
    e=np.array([
        [1-b],
        [b]
    ])
    res=np.matmul(c,d)
    res=np.matmul(res,e)
    return res

def warp_image(image,t,width,height):
    t=np.linalg.inv(t)
    result=np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            index=np.array([[i],[j],[1]])
            new_coordinate=np.matmul(t,index)
            new_coordinate=new_coordinate/new_coordinate[2,0]
            result[i,j]=bilinear_interpolation(image,new_coordinate[0,0],new_coordinate[1,0])
    return result




# def calc_distance(p1,p2):
#     res=np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
#     return res


# def calc_edge_length(book_coordinate):
#     edge_lenghts=np.zeros(4)
#     edge_lenghts[0]=calc_distance(book_coordinate[0],book_coordinate[1])
#     edge_lenghts[1]=calc_distance(book_coordinate[1],book_coordinate[2])
#     edge_lenghts[2]=calc_distance(book_coordinate[2],book_coordinate[3])
#     edge_lenghts[3]=calc_distance(book_coordinate[3],book_coordinate[0])
#     return edge_lenghts

start_time=time.time()
books=cv.imread("books.jpg")

extract_book(books,a_coordinate,190,350,'res16')
extract_book(books,b_coordinate,200,350,'res17')
extract_book(books,c_coordinate,230,420,'res18')
print(time.time()-start_time)



# print(calc_edge_length(a_coordinate))
# print(calc_edge_length(b_coordinate))
# print(calc_edge_length(c_coordinate))

# extract_book_opencv(books,a_coordinate,190,350,'A')
# extract_book_opencv(books,b_coordinate,200,350,'B')
# extract_book_opencv(books,c_coordinate,230,420,'C')


# t_a=cv.findHomography(a_coordinate,)


