import cv2 as cv
import imageio
import numpy as np
# import dlib
from scipy.spatial import Delaunay
import time


points_list1 = [[238, 463],
                [244, 540],
                [256, 616],
                [272, 689],
                [296, 760],
                [334, 826],
                [382, 883],
                [437, 930],
                [504, 942],
                [572, 932],
                [632, 887],
                [685, 833],
                [727, 771],
                [754, 700],
                [771, 626],
                [785, 550],
                [794, 472],
                [275, 454],
                [313, 423],
                [368, 419],
                [423, 431],
                [476, 451],
                [547, 450],
                [601, 431],
                [656, 421],
                [711, 425],
                [752, 455],
                [509, 489],
                [507, 541],
                [505, 595],
                [503, 649],
                [453, 674],
                [477, 684],
                [504, 693],
                [533, 684],
                [559, 674],
                [338, 482],
                [368, 464],
                [409, 465],
                [441, 494],
                [405, 503],
                [365, 502],
                [583, 494],
                [614, 465],
                [655, 464],
                [686, 482],
                [659, 502],
                [619, 503],
                [412, 773],
                [444, 757],
                [477, 747],
                [504, 756],
                [532, 748],
                [566, 758],
                [603, 774],
                [567, 802],
                [533, 816],
                [504, 819],
                [475, 816],
                [443, 802],
                [430, 774],
                [477, 773],
                [504, 777],
                [532, 773],
                [585, 775],
                [532, 772],
                [504, 776],
                [477, 772],
                [298, 246],
                [351, 238],
                [436, 264],
                [532, 270],
                [674, 239],
                [742, 257],
                [788, 361],
                [257, 329],
                [284, 272],
                [238, 441],
                [796, 438],
                [759, 292],
                [636, 262]
                ]
points2_list2 = [[217, 455],
                 [225, 535],
                 [240, 614],
                 [257, 690],
                 [286, 761],
                 [328, 826],
                 [382, 881],
                 [444, 925],
                 [514, 934],
                 [580, 921],
                 [632, 875],
                 [676, 820],
                 [711, 757],
                 [733, 688],
                 [747, 619],
                 [759, 547],
                 [762, 474],
                 [277, 444],
                 [317, 412],
                 [372, 405],
                 [427, 413],
                 [481, 432],
                 [558, 438],
                 [607, 420],
                 [656, 410],
                 [706, 416],
                 [738, 449],
                 [520, 482],
                 [522, 539],
                 [524, 596],
                 [526, 654],
                 [467, 673],
                 [494, 684],
                 [523, 694],
                 [549, 684],
                 [572, 672],
                 [341, 479],
                 [371, 463],
                 [409, 463],
                 [439, 486],
                 [406, 494],
                 [368, 494],
                 [584, 489],
                 [615, 466],
                 [652, 465],
                 [681, 482],
                 [654, 496],
                 [618, 496],
                 [419, 765],
                 [458, 753],
                 [494, 744],
                 [522, 752],
                 [548, 743],
                 [579, 750],
                 [609, 762],
                 [582, 793],
                 [552, 808],
                 [523, 812],
                 [493, 810],
                 [457, 795],
                 [436, 767],
                 [493, 767],
                 [522, 770],
                 [549, 766],
                 [593, 764],
                 [550, 768],
                 [523, 773],
                 [494, 769],
                 [291, 227],
                 [350, 222],
                 [438, 249],
                 [532, 255],
                 [657, 227],
                 [712, 242],
                 [763, 351],
                 [245, 310],
                 [277, 254],
                 [217, 423],
                 [768, 427],
                 [736, 281],
                 [627, 250]
                 ]

# predictor_path = 'shape_predictor_81_face_landmarks.dat'
#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)

start_time = time.time()
image1 = cv.imread("res01.png")
image2 = cv.imread("res02.png")


def get_points(image, name):
    # rect1 = detector(image, 1)[0]
    # landmarks1 = predictor(image, rect1)
    landmarks = []
    # for i in range(81):
    #     x = landmarks1.part(i).x
    #     y = landmarks1.part(i).y
    #     landmarks.append([x, y])

    # with open(f"points_{name}.txt", 'w') as f:
    #     for item in landmarks:
    #         f.write("%s,\n" % item)

    if name == 'image1':
        landmarks.extend(points_list1)
    else:
        landmarks.extend(points2_list2)

    landmarks.append([0, 0])
    landmarks.append([0, image.shape[1] - 1])
    landmarks.append([0, (image.shape[1] - 1) // 2])
    landmarks.append([(image.shape[0] - 1) // 2, 0])
    landmarks.append([(image.shape[0] - 1), 0])
    landmarks.append([(image.shape[0] - 1), (image.shape[1] - 1) // 2])
    landmarks.append([(image.shape[0] - 1) // 2, (image.shape[1] - 1)])
    landmarks.append([(image.shape[0] - 1), (image.shape[1] - 1)])
    if name == "image1":
        landmarks.append([201, 355])
        landmarks.append([223, 525])
        landmarks.append([209, 453])
        landmarks.append([859, 371])
        landmarks.append([817, 543])
        landmarks.append([861, 443])
    else:
        landmarks.append([169, 417])
        landmarks.append([205, 579])
        landmarks.append([179, 509])
        landmarks.append([791, 421])
        landmarks.append([761, 585])
        landmarks.append([777, 507])
    # image_copy = image.copy()
    # for i in landmarks:
    #     cv.circle(image_copy, (i[0], i[1]), 10, (0, 0, 255), -1)
    # cv.imwrite(f"{name}.jpg", image_copy)

    return landmarks


def get_triangles(points):
    tri = Delaunay(points)
    triangles = tri.simplices
    print(triangles.shape)
    return triangles


def draw_triangles(image, triangles, points):
    for i in range(triangles.shape[0]):
        cv.line(image, points[triangles[i, 0]], points[triangles[i, 1]], (0, 0, 0), 2)
        cv.line(image, points[triangles[i, 0]], points[triangles[i, 2]], (0, 0, 0), 2)
        cv.line(image, points[triangles[i, 1]], points[triangles[i, 2]], (0, 0, 0), 2)


def get_points_in_frame_t(points1, points2, t):
    result = (1 - t) * points1 + t * points2
    return result


def morph_two_image(image1, image2, result, points1, points2, points3, triangles, t):
    print("start")
    # result1 = np.zeros_like(image1)
    # result2 = np.zeros_like(image2)
    for i in range(triangles.shape[0]):
        # print(type(points1[triangles[i,0]].tolist()))
        # triangle1 = np.float32(
        #     [points1[triangles[i, 0]], points1[triangles[i, 1]], points1[triangles[i, 2]]])
        # triangle2 = np.float32(
        #     [points2[triangles[i, 0]], points2[triangles[i, 1]], points2[triangles[i, 2]]])
        # triangle3 = np.float32(
        #     [points3[triangles[i, 0]], points3[triangles[i, 1]], points3[triangles[i, 2]]])
        p1a = points1[triangles[i, 0]].astype('int32')
        p1b = points1[triangles[i, 1]].astype('int32')
        p1c = points1[triangles[i, 2]].astype('int32')
        p2a = points2[triangles[i, 0]].astype('int32')
        p2b = points2[triangles[i, 1]].astype('int32')
        p2c = points2[triangles[i, 2]].astype('int32')
        p3a = points3[triangles[i, 0]].astype('int32')
        p3b = points3[triangles[i, 1]].astype('int32')
        p3c = points3[triangles[i, 2]].astype('int32')

        minx1 = min(p1a[1], p1b[1], p1c[1])
        maxx1 = max(p1a[1], p1b[1], p1c[1])
        miny1 = min(p1a[0], p1b[0], p1c[0])
        maxy1 = max(p1a[0], p1b[0], p1c[0])

        minx2 = min(p2a[1], p2b[1], p2c[1])
        maxx2 = max(p2a[1], p2b[1], p2c[1])
        miny2 = min(p2a[0], p2b[0], p2c[0])
        maxy2 = max(p2a[0], p2b[0], p2c[0])

        minx3 = min(p3a[1], p3b[1], p3c[1])
        maxx3 = max(p3a[1], p3b[1], p3c[1])
        miny3 = min(p3a[0], p3b[0], p3c[0])
        maxy3 = max(p3a[0], p3b[0], p3c[0])

        new_x1a_coordinate = p1a[1] - minx1
        new_y1a_coordinate = p1a[0] - miny1
        new_x1b_coordinate = p1b[1] - minx1
        new_y1b_coordinate = p1b[0] - miny1
        new_x1c_coordinate = p1c[1] - minx1
        new_y1c_coordinate = p1c[0] - miny1

        new_x2a_coordinate = p2a[1] - minx2
        new_y2a_coordinate = p2a[0] - miny2
        new_x2b_coordinate = p2b[1] - minx2
        new_y2b_coordinate = p2b[0] - miny2
        new_x2c_coordinate = p2c[1] - minx2
        new_y2c_coordinate = p2c[0] - miny2

        new_x3a_coordinate = p3a[1] - minx3
        new_y3a_coordinate = p3a[0] - miny3
        new_x3b_coordinate = p3b[1] - minx3
        new_y3b_coordinate = p3b[0] - miny3
        new_x3c_coordinate = p3c[1] - minx3
        new_y3c_coordinate = p3c[0] - miny3

        new_triangle1 = np.array([[new_y1a_coordinate, new_x1a_coordinate],
                                  [new_y1b_coordinate, new_x1b_coordinate],
                                  [new_y1c_coordinate, new_x1c_coordinate]]).astype("float32")

        new_triangle2 = np.array([[new_y2a_coordinate, new_x2a_coordinate],
                                  [new_y2b_coordinate, new_x2b_coordinate],
                                  [new_y2c_coordinate, new_x2c_coordinate]]).astype("float32")

        new_triangle3 = np.array([[new_y3a_coordinate, new_x3a_coordinate],
                                  [new_y3b_coordinate, new_x3b_coordinate],
                                  [new_y3c_coordinate, new_x3c_coordinate]]).astype("float32")

        image1_rect = image1[minx1:maxx1 + 1, miny1:maxy1 + 1]
        image2_rect = image2[minx2:maxx2 + 1, miny2:maxy2 + 1]

        mask_width = maxy3 - miny3 + 1
        mask_height = maxx3 - minx3 + 1
        mask_height = mask_height
        mask_width = mask_width
        # print(mask_height)
        # print(mask_width)
        mask = np.zeros((mask_height, mask_width, 3))
        cv.fillConvexPoly(mask, new_triangle3.astype('int32'), (1, 1, 1))

        matrix1 = cv.getAffineTransform(new_triangle1, new_triangle3)
        new_image1_rect = cv.warpAffine(image1_rect, matrix1, (mask_width, mask_height))

        matrix2 = cv.getAffineTransform(new_triangle2, new_triangle3)
        new_image2_rect = cv.warpAffine(image2_rect, matrix2, (mask_width, mask_height))

        result_rect = (1 - t) * new_image1_rect + t * new_image2_rect

        result[minx3:maxx3 + 1, miny3:maxy3 + 1] = mask * result_rect + (1 - mask) * result[minx3:maxx3 + 1,
                                                                                     miny3:maxy3 + 1]


    print("End")
    return result


def image_morphing(image1, image2):
    points1 = get_points(image1, "image1")
    points1.append([201, 355])
    points1.append([223, 525])
    points1.append([209, 453])
    points1.append([859, 371])
    points1.append([817, 543])
    points1.append([861, 443])
    #####
    points1.append([139, 865])
    points1.append([859, 971])
    points1.append([933, 1013])
    points1.append([13, 901])
    points1.append([269, 813])
    points1.append([735, 891])
    points1.append([861, 179])
    points1.append([203, 179])
    points1.append([513, 77])
    points1.append([201, 845])
    points1.append([805, 933])
    points1 = np.array(points1)
    points2 = get_points(image2, "image2")
    points2.append([169, 417])
    points2.append([205, 579])
    points2.append([179, 509])
    points2.append([791, 421])
    points2.append([761, 585])
    points2.append([777, 507])
    points2.append([159, 859])
    points2.append([867, 861])
    points2.append([1007, 887])
    points2.append([17, 803])
    points2.append([277, 813])
    points2.append([651, 901])
    points2.append([789, 173])
    points2.append([157, 193])
    points2.append([513, 77])
    points2.append([209, 841])
    points2.append([745, 909])
    points2 = np.array(points2)
    triangles = get_triangles(points1)
    # image1_copy = image1.copy()
    # image2_copy = image2.copy()
    # draw_triangles(image1_copy, triangles, points1)
    # draw_triangles(image2_copy, triangles, points2)
    # cv.imwrite("bradpit_landmarks.jpg", image1_copy)
    # cv.imwrite("edawrd_landmarks.jpg", image2_copy)
    # print(triangles.shape)
    t = 0
    counter = 0
    images = []
    while t <= 1:
        result = np.zeros_like(image1)
        points3 = get_points_in_frame_t(points1, points2, t)
        # print(triangles)

        result = morph_two_image(image1, image2, result, points1, points2, points3, triangles, t)
        images.append(result)
        t += 1 / 45
        print(counter)
        counter += 1

    images_list2 = images.copy()
    images.reverse()
    images_list2.extend(images)
    images.extend(images_list2)
    print(time.time() - start_time)
    counter = 1
    for i in images_list2:
        video_writer.write(i)
        if counter == 15:
            cv.imwrite("res03.jpg", i)
        elif counter == 30:
            cv.imwrite("res04.jpg", i)
        counter += 1




frame_rate = 30
video_writer = cv.VideoWriter('morph.mp4', cv.VideoWriter_fourcc(*'XVID'),30,(image1.shape[1], image2.shape[0]))
image_morphing(image1, image2)

# cv.imwrite("bradpitt_landmarks.jpg", image1)
# cv.imwrite("edawrd_landmarks.jpg", image2)
video_writer.release()
print(time.time() - start_time)
