import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sy
import pandas as pd



def random_angle_generator(N):
    '''Generates a vector containing random angles ranging from 0 to pi
    input: N (Number of data to be generated)
    output: angle_array ( an array containing angles)'''

    angle_array = np.random.uniform(0, np.pi, size=N)
    return angle_array


def position_converter(angle_array, x_0, y_0):
    '''converts the generated angles into x-positions
    input: angle_array (an array containing angles), x_0 (position of light house in x), y_0 (height of light house)
    output: data_array (an array containing positions (data) corresponding to the angles in angle_array )'''

    data_array = np.zeros(len(angle_array))
    for i in range(len(angle_array)):
        if angle_array[i] == 0:
            angle_array[i] = 0.001
            data_array[i] = x_0 - y_0 / np.tan(angle_array[i])

        elif angle_array[i] == 2 * np.pi:
            angle_array[i] = np.pi - 0.001
            data_array = x_0 + abs(y_0 / (np.tan(angle_array[i]) - np.pi / 2))

        elif angle_array[i] == np.pi / 2:
            data_array[i] = x_0

        elif angle_array[i] > np.pi / 2 and angle_array[i] != np.pi:
            data_array[i] = x_0 + y_0 / (np.tan(angle_array[i] - np.pi / 2))

        elif angle_array[i] < np.pi / 2 and angle_array[i] != 0:
            data_array[i] = x_0 - y_0 / np.tan(angle_array[i])

    return data_array

def posterior(data_array, x, y):

    formula = 1

    for i in range(len(data_array)):
        formula = formula*(y/np.pi)*1/(y**2 + (data_array[i] - x)**2)

    return formula


def m_h(data_array, num_walk):
    x = 50
    y = 8
    mean = (x, y)
    cov = [[0.8, 0], [0, 0.8]]  # covariance matrix

    x_array = [0]*num_walk
    y_array = [0]*num_walk
    z_array = [0]*num_walk

    x_burn_array = [0] * (num_walk-700)
    y_burn_array = [0] * (num_walk-700)
    z_burn_array = [0] * (num_walk-700)

    for i in range(len(x_array)):
        U = np.random.random_sample()
        P_i = posterior(data_array, x, y)
        x_i, y_i = np.random.multivariate_normal(mean, cov).T
        P_f = posterior(data_array, x_i, y_i)
        r = P_f / P_i
        if U <= r:
            x_array[i] = x_i
            y_array[i] = y_i
            z_array[i] = P_f
            x = x_i
            y = y_i
            mean = (x_i, y_i)
        else:
            x_array[i] = x
            y_array[i] = y
            z_array[i] = P_i

    for i in range(len(x_burn_array)):
            x_burn_array[i] = x_array[i + 700]
            y_burn_array[i] = y_array[i + 700]
            z_burn_array[i] = z_array[i + 700]

    return x_array, y_array, x_burn_array, y_burn_array, z_burn_array


def marg_x(limit_x, x_burn_array, y_burn_array):

    marginal_x = np.zeros(np.size(limit_x))

    for i in range(len(limit_x)-1):
        for j in range(len(x_burn_array)):
            if x_burn_array[j] <= limit_x[i+1] and x_burn_array[j] > limit_x[i]:
                marginal_x[i] = marginal_x[i] + y_burn_array[j]

    return marginal_x

def marg_y(limit_y, x_burn_array, y_burn_array):

    marginal_y = np.zeros(np.size(limit_y))

    for i in range(len(limit_y)-1):
        for j in range(len(y_burn_array)):
            if y_burn_array[j] <= limit_y[i+1] and y_burn_array[j] > limit_y[i]:
                marginal_y[i] = marginal_y[i] + x_burn_array[j]

    return marginal_y






def light_house(x_0, y_0, N, shore_limit):
    '''light_house-function input parameters : (x_0,y_0,N)'''
    angle_array = random_angle_generator(N)
    data_array = position_converter(angle_array, x_0, y_0)

    num_walk = 3000  # num_walk = number of random walks
    x_array, y_array, x_burn_array, y_burn_array, z_burn_array = m_h(data_array, num_walk)

    start_limit_x = x_0 - 20
    start_limit_y = y_0 - 5
    interval = 0.5

    limit_x = np.arange(start_limit_x , x_0 + 20, interval)
    limit_y = np.arange(start_limit_y, y_0 + 5, interval)

    X, Y = np.meshgrid(limit_x, limit_y)


    marginal_x = marg_x(limit_x, x_burn_array, y_burn_array)
    marginal_y = marg_y(limit_y, x_burn_array, y_burn_array)

    '''making subplots'''


    plt.style.use('ggplot')

    plt.subplot(221)
    plt.plot(x_array, y_array, '.', color='m', markersize=1)
    #plt.contour(x_array, y_array)
    plt.ylabel('y')
    plt.title('MCMC (Gaussian proposal distribution) 2500 random walks', fontsize=10)
    plt.axis('equal')


    plt.subplot(224)
    plt.plot(limit_y, marginal_y)
    plt.xlabel('y')
    plt.title('Marginalization of x', fontsize=9)

    plt.subplot(223)
    plt.plot(limit_x, marginal_x)
    plt.xlabel('x')
    plt.title('Marginalization of y', fontsize=9)


    plt.subplot(222)
    plt.plot(x_burn_array, y_burn_array, '.', color='m', markersize=1)
    #plt.xlabel('x')
    plt.ylabel('y')
    plt.title('MCMC (Gaussian proposal distribution) with 1000 steps as burn-in period', fontsize=10)
    plt.axis('equal')

    plt.show()





#######################################################################################################################
# light_house-function input parameters : (x_0, y_0, N, shore_limit)
# x_0,y_0= position of light house, N = number of data, shore_limit = distance of the shore that we want to study

light_house(100, 10, 90, 400)







