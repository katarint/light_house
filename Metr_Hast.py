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
    cov = [[1.5, 0], [0, 1.5]]  # covariance matrix

    x_array = [0]*num_walk
    y_array = [0]*num_walk
    z_array = [0]*num_walk

    x_burn_array = [0] * (num_walk-800)
    y_burn_array = [0] * (num_walk-800)
    z_burn_array = [0] * (num_walk-800)

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
            x_burn_array[i] = x_array[i + 800]
            y_burn_array[i] = y_array[i + 800]
            z_burn_array[i] = z_array[i + 800]

    return x_array, y_array, x_burn_array, y_burn_array




def light_house(x_0, y_0, N, shore_limit):
    '''light_house-function input parameters : (x_0,y_0,N)'''
    angle_array = random_angle_generator(N)
    data_array = position_converter(angle_array, x_0, y_0)
    hist_range = [0, x_0 + 100]

    num_walk = 2500  # num_walk = number of random walks
    x_array, y_array, x_burn_array, y_burn_array = m_h(data_array, num_walk)

    '''making subplots'''


    plt.style.use('ggplot')

    plt.subplot(221)
    plt.plot(x_array, y_array, '.', color='m', markersize=1)
    #plt.contour(x_array, y_array)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('MCMC (Gaussian proposal distribution) 2500 random walks', fontsize=10)
    plt.axis('equal')

    plt.subplot(222)
    plt.plot(x_burn_array, y_burn_array, '.', color='m', markersize=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('MCMC (Gaussian proposal distribution) with 800 steps as burn-in period', fontsize=10)
    plt.axis('equal')
    plt.show()





#######################################################################################################################
# light_house-function input parameters : (x_0, y_0, N, shore_limit)
# x_0,y_0= position of light house, N = number of data, shore_limit = distance of the shore that we want to study

light_house(100, 10, 90, 400)







