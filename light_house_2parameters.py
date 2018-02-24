import numpy as np
import matplotlib.pyplot as plt
import math


def random_angle_generator(N):
    '''Generates a vector containing random angles ranging from 0 to pi
    input: N (Number of data to be generated)
    output: angle_array ( an array containing angles)'''

    angle_array = np.random.uniform(0, np.pi, size=N)
    return angle_array


def position_converter(angle_array, x_0, y_0):
    '''converts the generated angles into x-positions
    input: angle_array (an array containing angles), x_0 (position of light house in x), y_0 (height of light house)
    output: position_array (an array containing positions corresponding to the angles in angle_array)'''

    position_array = np.zeros(len(angle_array))
    for i in range(len(angle_array)):
        if angle_array[i] == 0:
            angle_array[i] = 0.001
            position_array[i] = x_0 - y_0 / np.tan(angle_array[i])

        elif angle_array[i] == 2 * np.pi:
            angle_array[i] = np.pi - 0.001
            position_array = x_0 + abs(y_0 / np.tan(angle_array[i]))

        elif angle_array[i] == np.pi / 2:
            angle_array[i] = np.pi / 2 - 0.001
            position_array[i] = x_0 - y_0 / np.tan(angle_array[i])

        elif angle_array[i] > np.pi / 2 and angle_array[i] != np.pi:
            position_array[i] = x_0 + abs(y_0 / np.tan(angle_array[i]))

        elif angle_array[i] < np.pi / 2 and angle_array[i] != 0:
            position_array[i] = x_0 - y_0 / np.tan(angle_array[i])

    return position_array


def posterior(position_array, x_0, y_0):
    '''calculates posterior for one parameter, x. When y is known'''
    # using a flat prior, the x_0 input is not necessary since an arbitrary limit/range can be used
    limit = np.arange(-100 + x_0, 100 + x_0)
    log_array = np.zeros((np.size(limit), np.size(position_array)))
    posterior_array = np.zeros(np.size(limit))

    for i in range(np.size(limit)):
        for k in range(0, np.size(position_array)):
            log_array[i][k] = - np.log(np.power(y_0, 2) + np.power(position_array[k] - (limit[i]), 2))

    for i in range(np.size(limit)):
        posterior_array[i] = np.sum(log_array[i])  # sums up all the log terms for a fixed alpha

    return posterior_array, limit

def log_posterior_2(x, y, position_array):

    Z = np.zeros((np.size(y), (np.size(x))))
    Z_temp = np.zeros(np.size(position_array))

    for j in np.arange(1, np.size(y)):
        for i in np.arange(np.size(x)):
            for k in np.arange(np.size(position_array)):
                Z_temp[k] = np.log(np.pi*(np.power(y[j], 2) + np.power(position_array[k] - x[i], 2)))
            Z[j][i] = np.size(position_array)*np.log(y[j]) - np.sum(Z_temp)


    return Z



def posterior_2(position_array, x_0, y_0):
    # the x_0 and y_0 input is not necessary since an arbitrary limit/range can be used
    limit_x = np.arange(-20 + x_0, 20 + x_0, 0.5)
    limit_y = np.arange(0, 20 + y_0, 0.5)
    x = limit_x
    y = limit_y
    X, Y = np.meshgrid(x, y)
    Z = log_posterior_2(x, y, position_array)

    return X, Y, Z



def light_house(x_0, y_0, N):
    '''light_house-function input parameters : (x_0,y_0,N)'''
    angle_array = random_angle_generator(N)
    position_array = position_converter(angle_array, x_0, y_0)
    hist_range = [x_0 - 100, x_0 + 100]

    posterior_array, limit = posterior(position_array, x_0, y_0)

    position_average = np.sum(position_array) / N

    X, Y, Z = posterior_2(position_array, x_0, y_0)

    '''making subplots'''

    '''Finding maximum value of posterior'''

    posterior_max = np.argmax(posterior_array)  # returns the index for the maximum value in posterior_array
    position_max = limit[posterior_max]  # finds the x-position for the best estimate

    plt.style.use('ggplot')

    plt.subplot(221)
    plt.hist(position_array, 'fd', hist_range, normed=False, weights=None, density=None)
    plt.title('Uniformly distributed data')
    plt.ylabel('count')

    plt.subplot(222)
    plt.plot(limit, posterior_array)
    plt.vlines(position_average, -2000, 100, colors='g')
    plt.title('Posterior for x when y is known')
    plt.xlabel('x')
    plt.ylabel('log (prob(x|y,I))')

    plt.legend((N, round(position_average)), loc='upper right', shadow=True, prop={'size': 10})


    plt.subplot(223)
    plt.contour(X, Y, Z, 70, colors='black')
    plt.title('Prob(x,y|{Data},I) contour plot', fontsize=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



#######################################################################################################################
# light_house-function input parameters : (x_0,y_0,N)
# x_0,y_0= position of light house, N = number of data

light_house(0, 10, 200)






