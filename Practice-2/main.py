import copy
from math import sqrt

from PIL import Image, ImageOps


def average_filtering(image, filter_size):
    border_size = int((3 - 1) / 2)
    image_copy = copy.deepcopy(image)
    divider_value = filter_size * filter_size

    image_with_border = ImageOps.expand(image_copy, border=border_size, fill=0)
    for i in range(border_size, image_with_border.width - (border_size * 2)):
        for j in range(border_size, image_with_border.height - (border_size * 2)):
            value = 0
            for i_filter in range(i - border_size, i - border_size + filter_size):
                for j_filter in range(j - border_size, j - border_size + filter_size):
                    value += int(image_with_border.getpixel((i_filter, j_filter)) / divider_value)

            image_copy.putpixel((i, j), value)

    return image_copy


def weight_avg_filtering(image, filter_size):
    border_size = int((filter_size - 1) / 2)
    image_copy = copy.deepcopy(image)

    filter = generate_filter_matrix(filter_size)
    convoluted_filter = convolute_matrix(filter)
    image_with_border = ImageOps.expand(image_copy, border=border_size, fill=0)
    for i in range(border_size, image_with_border.width - (border_size * 2)):
        for j in range(border_size, image_with_border.height - (border_size * 2)):
            value = 0
            filter_i_index = 0
            filter_j_index = 0
            for i_filter in range(i - border_size, i - border_size + filter_size):
                for j_filter in range(j - border_size, j - border_size + filter_size):
                    multiplayer = convoluted_filter[filter_i_index][filter_j_index]
                    value += int(image_with_border.getpixel((i_filter, j_filter)) * multiplayer)
                    filter_j_index += 1
                filter_j_index = 0
                filter_i_index += 1

            image_copy.putpixel((i, j), value)

    return image_copy


def sobel_filtering(image, direction_axis):
    border_size = 1
    image_copy = copy.deepcopy(image)

    if direction_axis == 0:
        filter = generate_filter_matrix_sobel(0)
    else:
        filter = generate_filter_matrix_sobel(1)

    convoluted_filter = convolute_matrix(filter)
    image_with_border = ImageOps.expand(image_copy, border=border_size, fill=0)

    for i in range(border_size, image_with_border.width - (border_size * 2)):
        for j in range(border_size, image_with_border.height - (border_size * 2)):
            value = 0
            filter_i_index = 0
            filter_j_index = 0
            for i_filter in range(i - border_size, i - border_size + 3):
                for j_filter in range(j - border_size, j - border_size + 3):
                    multiplayer = convoluted_filter[filter_i_index][filter_j_index]
                    value += int(image_with_border.getpixel((i_filter, j_filter)) * multiplayer)
                    filter_j_index += 1
                filter_j_index = 0
                filter_i_index += 1

            image_copy.putpixel((i, j), value)

    return image_copy


def laplacian_filtering(image):
    border_size = 1
    image_copy = copy.deepcopy(image)
    filter = generate_filter_matrix_laplacian()

    convoluted_filter = convolute_matrix(filter)
    image_with_border = ImageOps.expand(image_copy, border=border_size, fill=0)

    for i in range(border_size, image_with_border.width - (border_size * 2)):
        for j in range(border_size, image_with_border.height - (border_size * 2)):
            value = 0
            filter_i_index = 0
            filter_j_index = 0
            for i_filter in range(i - border_size, i - border_size + 3):
                for j_filter in range(j - border_size, j - border_size + 3):
                    multiplayer = convoluted_filter[filter_i_index][filter_j_index]
                    value += int(image_with_border.getpixel((i_filter, j_filter)) * multiplayer)
                    filter_j_index += 1
                filter_j_index = 0
                filter_i_index += 1

            image_copy.putpixel((i, j), value)

    return image_copy


def edge_detector(image, gradiant_kernel, threshold):
    border_size = 1
    image_copy_threshold = copy.deepcopy(image)
    image_copy_x_derivative = copy.deepcopy(image)
    image_copy_y_derivative = copy.deepcopy(image)

    if gradiant_kernel == 0:
        filter_x = generate_filter_matrix_sobel(0)
        filter_y = generate_filter_matrix_sobel(1)
    else:
        filter_x = generate_filter_matrix_sobel_version_2(0)
        filter_y = generate_filter_matrix_sobel_version_2(1)

    convoluted_filter_x = convolute_matrix(filter_x)
    convoluted_filter_y = convolute_matrix(filter_y)
    image_with_border = ImageOps.expand(image_copy_threshold, border=border_size, fill=0)
    for i in range(border_size, image_with_border.width - (border_size * 2)):
        for j in range(border_size, image_with_border.height - (border_size * 2)):
            value_x = 0
            value_y = 0
            filter_i_index = 0
            filter_j_index = 0
            for i_filter in range(i - border_size, i - border_size + 3):
                for j_filter in range(j - border_size, j - border_size + 3):
                    multiplayer_x = convoluted_filter_x[filter_i_index][filter_j_index]
                    multiplayer_y = convoluted_filter_y[filter_i_index][filter_j_index]
                    value_x += int(image_with_border.getpixel((i_filter, j_filter)) * multiplayer_x)
                    value_y += int(image_with_border.getpixel((i_filter, j_filter)) * multiplayer_y)
                    filter_j_index += 1
                filter_j_index = 0
                filter_i_index += 1

            image_copy_x_derivative.putpixel((i, j), value_x)
            image_copy_y_derivative.putpixel((i, j), value_y)
            magnitude = int(sqrt(pow(value_x, 2) + pow(value_y, 2)))

            if magnitude >= threshold:
                image_copy_threshold.putpixel((i, j), 255)
            else:
                image_copy_threshold.putpixel((i, j), 0)

    return [image_copy_x_derivative, image_copy_y_derivative, image_copy_threshold]


def convolute_matrix(filter_matrix):
    convoluted_matrix = []
    j_normal = 0
    for i, value_i in reversed(list(enumerate(filter_matrix))):
        row = [int] * len(filter_matrix)
        for j, value_j in reversed(list(enumerate(filter_matrix[i]))):
            row[j_normal] = filter_matrix[i][j]
            j_normal += 1
        convoluted_matrix.append(row)
        j_normal = 0

    return convoluted_matrix


def generate_filter_matrix_sobel(direction_axis):
    filter_matrix = []

    if direction_axis == 0:
        filter_matrix_row_one = [-1, 0, 1]
        filter_matrix_row_two = [-2, 0, 2]
        filter_matrix_row_three = [-1, 0, 1]

        filter_matrix.append(filter_matrix_row_one)
        filter_matrix.append(filter_matrix_row_two)
        filter_matrix.append(filter_matrix_row_three)

    else:
        filter_matrix_row_one = [-1, -2, -1]
        filter_matrix_row_two = [0, 0, 0]
        filter_matrix_row_three = [1, 2, 1]

        filter_matrix.append(filter_matrix_row_one)
        filter_matrix.append(filter_matrix_row_two)
        filter_matrix.append(filter_matrix_row_three)

    return filter_matrix


def generate_filter_matrix_sobel_version_2(direction_axis):
    filter_matrix = []

    if direction_axis == 0:
        filter_matrix_row_one = [-1, 0, 1]
        filter_matrix_row_two = [-1, 0, 1]
        filter_matrix_row_three = [-1, 0, 1]

        filter_matrix.append(filter_matrix_row_one)
        filter_matrix.append(filter_matrix_row_two)
        filter_matrix.append(filter_matrix_row_three)

    else:
        filter_matrix_row_one = [-1, -1, -1]
        filter_matrix_row_two = [0, 0, 0]
        filter_matrix_row_three = [1, 1, 1]

        filter_matrix.append(filter_matrix_row_one)
        filter_matrix.append(filter_matrix_row_two)
        filter_matrix.append(filter_matrix_row_three)

    return filter_matrix


def generate_filter_matrix_laplacian():
    filter_matrix = []

    filter_matrix_row_one = [0, -1, 0]
    filter_matrix_row_two = [-1, 4, -1]
    filter_matrix_row_three = [0, -1, 0]

    filter_matrix.append(filter_matrix_row_one)
    filter_matrix.append(filter_matrix_row_two)
    filter_matrix.append(filter_matrix_row_three)

    return filter_matrix


def generate_filter_matrix(filter_size):
    filter_matrix = []

    if filter_size == 3:
        filter_matrix_row_one = [1 / 16, 2 / 16, 1 / 16]
        filter_matrix_row_two = [2 / 16, 4 / 16, 2 / 16]
        filter_matrix_row_three = [1 / 16, 2 / 16, 1 / 16]

        filter_matrix.append(filter_matrix_row_one)
        filter_matrix.append(filter_matrix_row_two)
        filter_matrix.append(filter_matrix_row_three)

    elif filter_size == 5:
        filter_matrix_row_one = [2 / 115, 4 / 115, 5 / 115, 4 / 115, 2 / 115]
        filter_matrix_row_two = [4 / 115, 9 / 115, 12 / 115, 9 / 115, 4 / 115]
        filter_matrix_row_three = [5 / 115, 12 / 115, 15 / 115, 12 / 115, 5 / 115]
        filter_matrix_row_four = [4 / 115, 9 / 115, 12 / 115, 9 / 115, 4 / 115]
        filter_matrix_row_five = [2 / 115, 4 / 115, 5 / 115, 4 / 115, 2 / 115]

        filter_matrix.append(filter_matrix_row_one)
        filter_matrix.append(filter_matrix_row_two)
        filter_matrix.append(filter_matrix_row_three)
        filter_matrix.append(filter_matrix_row_four)
        filter_matrix.append(filter_matrix_row_five)

    return filter_matrix


if __name__ == '__main__':
    image = Image.open('img/imagen_CNN_reduced.jpg')
    greyscale_image = ImageOps.grayscale(image)

    # Exercise 1
    result_image_sobel_filter_X = sobel_filtering(greyscale_image, 0)
    result_image_sobel_filter_X.save('img/sobelFilter/result_image_sobel_filter_X.jpg')
    result_image_sobel_filter_Y = sobel_filtering(greyscale_image, 1)
    result_image_sobel_filter_Y.save('img/sobelFilter/result_image_sobel_filter_Y.jpg')

    # Exercise 2
    result_image_laplacian_filter = laplacian_filtering(greyscale_image)
    result_image_laplacian_filter.save('img/laplacianFilter/result_image_laplacian_filter.jpg')

    # Exercise 3
    # Filter 0
    result_edge_detector_filter_list = edge_detector(greyscale_image, 0, 180)
    result_edge_detector_filter_list[0].save('img/edgeDetector/filter_0/result_edge_detector_filter_x.jpg')
    result_edge_detector_filter_list[1].save('img/edgeDetector/filter_0/result_edge_detector_filter_y.jpg')
    result_edge_detector_filter_list[2].save('img/edgeDetector/filter_0/result_edge_detector_filter_threshold.jpg')

    # Filter 1
    result_edge_detector_filter_list = edge_detector(greyscale_image, 1, 180)
    result_edge_detector_filter_list[0].save('img/edgeDetector/filter_1/result_edge_detector_filter_x.jpg')
    result_edge_detector_filter_list[1].save('img/edgeDetector/filter_1/result_edge_detector_filter_y.jpg')
    result_edge_detector_filter_list[2].save('img/edgeDetector/filter_1/result_edge_detector_filter_threshold.jpg')
