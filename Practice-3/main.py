import matplotlib.pyplot as plt
import numpy as np


def calculate_summation_theta_0(list_x_values, list_y_values, theta_0_parameter, theta_1_parameter):
    summation = 0
    for i in range(0, len(list_x_values)):
        x_i = list_x_values[i][0]
        y_i = list_y_values[i][0]
        summation += (theta_0_parameter + (theta_1_parameter * x_i)) - y_i

    return summation


def calculate_summation_theta_1(list_x_values, list_y_values, theta_0_parameter, theta_1_parameter):
    summation = 0
    for i in range(0, len(list_x_values)):
        x_i = list_x_values[i][0]
        y_i = list_y_values[i][0]
        summation += ((theta_0_parameter + (theta_1_parameter * x_i)) - y_i) * x_i

    return summation


if __name__ == '__main__':
    # read flash.dat to a list of lists
    dat_content_x = [i.strip().split() for i in open("data/ex2x.dat").readlines()]
    dat_content_y = [i.strip().split() for i in open("data/ex2y.dat").readlines()]
    for i in range(0, len(dat_content_x)):
        dat_content_x[i][0] = float(dat_content_x[i][0])
        dat_content_y[i][0] = float(dat_content_y[i][0])

    plt.scatter(dat_content_x, dat_content_y)
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.title('Practice 3')
    plt.show()

    theta_0 = 0
    theta_1 = 0
    alpha = 0.07
    convergence = False
    m = len(dat_content_x)

    summation_0 = calculate_summation_theta_0(dat_content_x, dat_content_y, theta_0, theta_1)
    summation_1 = calculate_summation_theta_1(dat_content_x, dat_content_y, theta_0, theta_1)

    temp_theta_0 = theta_0 - (alpha * (1 / m) * summation_0)
    temp_theta_1 = theta_1 - (alpha * (1 / m) * summation_1)

    theta_0 = temp_theta_0
    theta_1 = temp_theta_1

    print("Value of Theta_0 first iteration: ", "{:.3f}".format(theta_0))
    print("Value of Theta_1 first iteration: ", "{:.3f}".format(theta_1))
    print()

    while not convergence:
        summation_0 = calculate_summation_theta_0(dat_content_x, dat_content_y, theta_0, theta_1)
        summation_1 = calculate_summation_theta_1(dat_content_x, dat_content_y, theta_0, theta_1)

        temp_theta_0 = theta_0 - (alpha * (1 / m) * summation_0)
        temp_theta_1 = theta_1 - (alpha * (1 / m) * summation_1)

        if abs(theta_0 - temp_theta_0) < 0.0001 and abs(theta_1 - temp_theta_1) < 0.0001:
            convergence = True

        theta_0 = temp_theta_0
        theta_1 = temp_theta_1

    print("Value of Theta_0 at end: ", "{:.3f}".format(theta_0))
    print("Value of Theta_1 at end: ", "{:.3f}".format(theta_1))
    print()

    plt.scatter(dat_content_x, dat_content_y)
    x = np.arange(2, 9)
    y = theta_0 + theta_1 * x
    plt.plot(x, y)
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.title('Practice 3')
    plt.show()

    prediction_first_boy_height = theta_0 + theta_1 * 3.5
    prediction_second_boy_height = theta_0 + theta_1 * 7

    print("The 3.5 year old boy must have a height of: ", "{:.3f}".format(prediction_first_boy_height))
    print("The 7.0 year old boy must have a height of: ", "{:.3f}".format(prediction_second_boy_height))

    plt.scatter(dat_content_x, dat_content_y)
    x = np.arange(2, 9)
    y = theta_0 + theta_1 * x
    plt.plot(x, y)
    plt.scatter([[3.5], [7]], [[prediction_first_boy_height], [prediction_second_boy_height]], color='red')
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.title('Practice 3')
    plt.show()
