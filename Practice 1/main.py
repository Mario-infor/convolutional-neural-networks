import copy
from PIL import Image, ImageOps


def median_filtering(image, filter_size):

    border_size = int((filter_size-1)/2)
    image_copy = copy.deepcopy(image)

    image_with_border = ImageOps.expand(image_copy, border=border_size, fill=0)
    for i in range(border_size, image_with_border.width - (border_size * 2)):
        for j in range(border_size, image_with_border.height - (border_size * 2)):
            list_elements = []
            for i_filter in range(i-border_size, i-border_size + filter_size):
                for j_filter in range(j-border_size, j-border_size + filter_size):
                    list_elements.append(image_with_border.getpixel((i_filter, j_filter)))

            list_elements.sort()
            media = list_elements[int(len(list_elements) / 2)]

            image_copy.putpixel((i, j), media)

    return image_copy


def average_filtering(image, filter_size):

    border_size = int((filter_size-1)/2)
    image_copy = copy.deepcopy(image)
    divider_value = filter_size*filter_size

    image_with_border = ImageOps.expand(image_copy, border=border_size, fill=0)
    for i in range(border_size, image_with_border.width - (border_size * 2)):
        for j in range(border_size, image_with_border.height - (border_size * 2)):
            value = 0
            for i_filter in range(i-border_size, i-border_size + filter_size):
                for j_filter in range(j-border_size, j-border_size + filter_size):
                    value += int(image_with_border.getpixel((i_filter, j_filter)) / divider_value)

            image_copy.putpixel((i, j), value)

    return image_copy


def weight_avg_filtering(image, filter_size):

    border_size = int((filter_size-1)/2)
    image_copy = copy.deepcopy(image)

    filter = generate_filter_matrix(filter_size)
    convoluted_filter = convolution_matrix(filter)
    image_with_border = ImageOps.expand(image_copy, border=border_size, fill=0)
    for i in range(border_size, image_with_border.width - (border_size * 2)):
        for j in range(border_size, image_with_border.height - (border_size * 2)):
            value = 0
            filter_i_index = 0
            filter_j_index = 0
            for i_filter in range(i-border_size, i-border_size + filter_size):
                for j_filter in range(j-border_size, j-border_size + filter_size):
                    multiplayer = convoluted_filter[filter_i_index][filter_j_index]
                    value += int(image_with_border.getpixel((i_filter, j_filter)) * multiplayer)
                    filter_j_index += 1
                filter_j_index = 0
                filter_i_index += 1

            image_copy.putpixel((i, j), value)

    return image_copy


def avg_color_filtering(image, filter_size):
    border_size = int((filter_size - 1) / 2)
    image_copy = copy.deepcopy(image)

    if filter_size == 3:
        divider_value = 27
    else:
        divider_value = filter_size * filter_size * 3

    image_with_border = ImageOps.expand(image_copy, border=border_size, fill=0)
    for i in range(border_size, image_with_border.width - (border_size * 2)):
        for j in range(border_size, image_with_border.height - (border_size * 2)):
            value = 0
            for i_filter in range(i - border_size, i - border_size + filter_size):
                for j_filter in range(j - border_size, j - border_size + filter_size):
                    temp_pixel = image_with_border.getpixel((i_filter, j_filter))
                    value += int(temp_pixel[0] / divider_value)
                    value += int(temp_pixel[1] / divider_value)
                    value += int(temp_pixel[2] / divider_value)

            image_copy.putpixel((i, j), (value, value, value))

    return image_copy


def generate_filter_matrix(filter_size):
    filter_matrix = []

    if filter_size == 3:
        filter_matrix_row_one = [1/16, 2/16, 1/16]
        filter_matrix_row_two = [2/16, 4/16, 2/16]
        filter_matrix_row_three = [1/16, 2/16, 1/16]

        filter_matrix.append(filter_matrix_row_one)
        filter_matrix.append(filter_matrix_row_two)
        filter_matrix.append(filter_matrix_row_three)

    elif filter_size == 5:
        filter_matrix_row_one = [2/115, 4/115, 5/115, 4/115, 2/115]
        filter_matrix_row_two = [4/115, 9/115, 12/115, 9/115, 4/115]
        filter_matrix_row_three = [5/115, 12/115, 15/115, 12/115, 5/115]
        filter_matrix_row_four = [4/115, 9/115, 12/115, 9/115, 4/115]
        filter_matrix_row_five = [2/115, 4/115, 5/115, 4/115, 2/115]

        filter_matrix.append(filter_matrix_row_one)
        filter_matrix.append(filter_matrix_row_two)
        filter_matrix.append(filter_matrix_row_three)
        filter_matrix.append(filter_matrix_row_four)
        filter_matrix.append(filter_matrix_row_five)

    return filter_matrix


def convolution_matrix(filter_matrix):
    convoluted_matrix = []
    j_normal = 0
    for i, value_i in reversed(list(enumerate(filter_matrix))):
        row = [int]*len(filter_matrix)
        for j, value_j in reversed(list(enumerate(filter_matrix[i]))):
            row[j_normal] = filter_matrix[i][j]
            j_normal += 1
        convoluted_matrix.append(row)
        j_normal = 0

    return convoluted_matrix


if __name__ == '__main__':
    image = Image.open('img/imagen_tarea_CNN_reducida.jpg')
    greyscale_image = ImageOps.grayscale(image)

    # Exercise 1
    new_image_media_filter_3 = median_filtering(greyscale_image, 3)
    new_image_media_filter_3.save('img/medianFiltering/result_median_filter_3.jpg')
    new_image_media_filter_5 = median_filtering(greyscale_image, 5)
    new_image_media_filter_5.save('img/medianFiltering/result_median_filter_5.jpg')
    new_image_media_filter_7 = median_filtering(greyscale_image, 7)
    new_image_media_filter_7.save('img/medianFiltering/result_median_filter_7.jpg')

    # Exercise 2
    new_image_average_filter_3 = average_filtering(greyscale_image, 3)
    new_image_average_filter_3.save('img/averageFiltering/new_image_average_filter_3.jpg')
    new_image_average_filter_5 = average_filtering(greyscale_image, 5)
    new_image_average_filter_5.save('img/averageFiltering/new_image_average_filter_5.jpg')
    new_image_average_filter_7 = average_filtering(greyscale_image, 7)
    new_image_average_filter_7.save('img/averageFiltering/new_image_average_filter_7.jpg')

    # Exercise 3
    new_image_weight_avg_filter_3 = weight_avg_filtering(greyscale_image, 3)
    new_image_weight_avg_filter_3.save('img/weightavgFiltering/new_image_weight_avg_filter_3.jpg')
    new_image_weight_avg_filter_5 = weight_avg_filtering(greyscale_image, 5)
    new_image_weight_avg_filter_5.save('img/weightavgFiltering/new_image_weight_avg_filter_5.jpg')

    # Exercise 4
    new_image_avg_color_filter_3 = avg_color_filtering(image, 3)
    new_image_avg_color_filter_3.save('img/avgColorFiltering/new_image_avg_color_filter_3.jpg')
    new_image_avg_color_filter_5 = avg_color_filtering(image, 5)
    new_image_avg_color_filter_5.save('img/avgColorFiltering/new_image_avg_color_filter_5.jpg')
