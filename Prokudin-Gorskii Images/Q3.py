import cv2 as cv2
import numpy as np
import time


def calc_start_end(image, blue, green, red, axis, shape_index):
    count_red = np.count_nonzero(red <= 80, axis=axis)
    count_blue = np.count_nonzero(blue <= 80, axis=axis)
    count_green = np.count_nonzero(green <= 80, axis=axis)
    a = np.arange(image.shape[shape_index])
    red_selected = a[np.where(count_red > (0.6 * image.shape[shape_index]))]
    blue_selected = a[np.where(count_blue > (0.6 * image.shape[shape_index]))]
    green_selected = a[np.where(count_green > (0.6 * image.shape[shape_index]))]
    x = int(image.shape[shape_index] * 0.1)
    y = int(image.shape[shape_index] * 0.9)
    start_row_red = red_selected[np.where(red_selected < x)]
    end_row_red = red_selected[np.where(red_selected > y)]

    start_row_blue = blue_selected[np.where(blue_selected < x)]
    end_row_blue = blue_selected[np.where(blue_selected > y)]

    start_row_green = green_selected[np.where(green_selected < x)]
    end_row_green = green_selected[np.where(green_selected > y)]

    start_union_row = np.union1d(start_row_red, start_row_blue)
    start_union_row = np.union1d(start_union_row, start_row_green)

    end_union_row = np.union1d(end_row_red, end_row_blue)
    end_union_row = np.union1d(end_union_row, end_row_green)

    start_row_index = np.max(start_union_row)
    end_row_index = np.min(end_union_row)
    return start_row_index, end_row_index


def auto_crop(image):
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    start_row_index, end_row_index = calc_start_end(image, blue, green, red, 1, 0)
    start_col_index, end_col_index = calc_start_end(image, blue, green, red, 0, 1)
    image = image[start_row_index:end_row_index, start_col_index:end_col_index]
    return image


def translation(image, tx, ty):
    new_image = np.roll(image, tx, axis=1)
    new_image = np.roll(new_image, ty, axis=0)
    if tx > 0:
        new_image[:, :tx] = 0
    elif tx < 0:
        new_image[:, (new_image.shape[1] + tx):] = 0
    if ty > 0:
        new_image[:ty, :] = 0
    elif ty < 0:
        new_image[(new_image.shape[0] + ty):, :] = 0
    return new_image


def image_pyramids(image, number_layer):
    result_list = []
    result_list.append(image)
    for i in range(1, number_layer):
        new_image = cv2.GaussianBlur(result_list[i - 1], (7, 7), 0)
        new_image = new_image[:, 0::2]
        new_image = new_image[0::2, :]
        result_list.append(new_image)

    return result_list


def find_tx_ty_all_layers(image_pyramid, template_pyramid, s):
    start_h = -template_pyramid[-1].shape[0] // 2
    stop_h = template_pyramid[-1].shape[0] // 2
    start_w = -template_pyramid[-1].shape[1] // 2
    stop_w = template_pyramid[-1].shape[1] // 2
    tx_list = []
    ty_list = []
    tx, ty = find_tx_ty_each_layer(image_pyramid[-1], template_pyramid[-1], start_h, stop_h, start_w, stop_w)
    tx_list.insert(0, tx)
    ty_list.insert(0, ty)
    print(f'level {len(image_pyramid)} tx:{tx}, ty:{ty}')
    for i in range(-2, -len(red_pyramid) - 1, -1):
        tx, ty = find_tx_ty_each_layer(image_pyramid[i], template_pyramid[i],
                                       2 * tx - s, 2 * tx + s, 2 * ty - s, 2 * ty + s)
        tx_list.insert(0, tx)
        ty_list.insert(0, ty)
        print(f'level {i + len(image_pyramid)} tx:{tx}, ty:{ty}')

    return tx, ty, tx_list, ty_list


def find_tx_ty_each_layer(image, template, start_h, stop_h, start_w, stop_w):
    h = np.arange(start_h, stop_h + 1)
    w = np.arange(start_w, stop_w + 1)
    minimum = ssd(image, translation(template, h[0], w[0]))
    min_i = 0
    min_j = 0
    for i in range(h.shape[0]):
        for j in range(w.shape[0]):
            a = ssd(image, translation(template, h[i], w[j]))
            if minimum > a:
                minimum = a
                min_i = i
                min_j = j

    return h[min_i], w[min_j]


def ssd(image, template):
    x = image - template
    x = x.astype('int64')
    x = (x * x)
    result = np.sum(x)
    return result


start_time = time.time()
image = cv2.imread("Amir.tif", cv2.IMREAD_UNCHANGED)

print(image.shape)
newH = image.shape[0] // 3
newH = newH * 3
image = image[:newH, :]
blue, green, red = np.vsplit(image, 3)
print(type(red[0, 0]))
print(red.shape)
print(blue.shape)
print(green.shape)

red = red.astype('int16')
blue = blue.astype('int16')
green = green.astype('int16')
red_pyramid = image_pyramids(red, 6)
green_pyramid = image_pyramids(green, 6)
blue_pyramid = image_pyramids(blue, 6)

# todo Blue
tx_green, ty_green, green_tx_list, green_ty_list = find_tx_ty_all_layers(blue_pyramid, green_pyramid, 4)
tx_red, ty_red, red_tx_list, red_ty_list = find_tx_ty_all_layers(blue_pyramid, red_pyramid, 4)
green = translation(green, tx_green, ty_green)
red = translation(red, tx_red, ty_red)

for i in range(len(red_pyramid)):
    green2 = translation(green_pyramid[i], green_tx_list[i], green_ty_list[i])
    red2 = translation(red_pyramid[i], red_tx_list[i], red_ty_list[i])
    blue2 = blue_pyramid[i]
    blue2 = (blue2 / 256).astype('uint8')
    red2 = (red2 / 256).astype('uint8')
    green2 = (green2 / 256).astype('uint8')
    mixed2 = np.zeros((red2.shape[0], red2.shape[1], 3))
    mixed2[:, :, 0] = blue2
    mixed2[:, :, 1] = green2
    mixed2[:, :, 2] = red2
    cv2.imwrite(f'pyramid{i}.jpg', mixed2)

blue = (blue / 256).astype('uint8')
red = (red / 256).astype('uint8')
green = (green / 256).astype('uint8')

mixed = np.zeros((red.shape[0], image.shape[1], 3))
mixed[:, :, 0] = blue
mixed[:, :, 1] = green
mixed[:, :, 2] = red

mixed = auto_crop(mixed)

cv2.imwrite("res03-Amir.jpg", mixed)
print(time.time() - start_time)
