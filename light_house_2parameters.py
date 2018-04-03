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


def posterior_2(data_array, x_0, y_0):
    limit_x = np.arange(0.5, x_0 + 100, 0.25)
    limit_y = np.arange(0.5, 10 + y_0, 0.25)
    x = limit_x
    y = limit_y
    X, Y = np.meshgrid(x, y)
    Z = log_posterior_2(x, y, data_array)

    for j in np.arange(1, np.size(y)):
        for i in np.arange(1, np.size(x)):
            Z[j][i] = np.exp(Z[j][i])

    return X, Y, Z, limit_x, limit_y

'''
def log_posterior_2(x, y, data_array):

    Z = np.zeros((np.size(y), (np.size(x))))  # Z: a matrix with ((#y-row,  #x-column))
    Z_temp = np.zeros(np.size(data_array))

    for j in np.arange(1, np.size(y)):
        for i in np.arange(1, np.size(x)):
            for k in np.arange(np.size(data_array)):
                Z_temp[k] = np.log((np.power(y[j], 2) + np.power(data_array[k] - x[i], 2)))
            Z[j][i] = np.size(data_array)*np.log(y[j]) - np.sum(Z_temp)

    return Z
'''

def log_posterior_2(x, y, data_array):
    '''Calculates posterior and takes the logarithm'''

    Z = np.zeros((np.size(y), (np.size(x))))  # Z: a matrix with ((#y-row,  #x-column))

    for i in np.arange(1, np.size(y)):
        for j in np.arange(1, np.size(x)):
            formula = np.size(data_array) * np.log(y[i] / np.pi)
            for k in np.arange(np.size(data_array)):
                formula = formula - np.log(y[i]**2 + (data_array[k] - x[j])**2)
            Z[i][j] = formula

    z_max = Z.max()

    for j in np.arange(np.size(y)):
        for i in np.arange(np.size(x)):
            Z[j][i] = Z[j][i] - z_max     # scaling down the z-axis

    return Z

def marginal_x(Z, limit_x, limit_y):

    marg_posterior_x = [0, ]*np.size(limit_x)

    for i in range(len(limit_x)):
        marg_posterior_x[i] = 0
        for j in range(len(limit_y)):
            marg_posterior_x[i] = marg_posterior_x[i] + Z[j][i]*0.25

    return marg_posterior_x


def marginal_y(Z, limit_x, limit_y):

    marg_posterior_y = [0, ]*np.size(limit_y)

    for i in range(len(limit_y)):
        marg_posterior_y[i] = 0
        for j in range(len(limit_x)):
            marg_posterior_y[i] = marg_posterior_y[i] + Z[i][j]*0.25

    return marg_posterior_y


'''
def marginal_x(data_array, x_0, y_0, shore_limit):

    limit_x = np.arange(0.5, x_0 + 50, 0.5)
    limit_y = np.arange(0.5, 10 + y_0, 0.5)
    prob_array = [0]*len(limit_y)
    marg_posterior_x = [0]*len(limit_x)

    for i in range(len(limit_x)):
        for j in range(len(limit_y)):
            formula = np.size(data_array)*np.log(limit_y[j])
            for n in range(len(data_array)):
                formula = formula - np.log(limit_y[j]**2 + (data_array[n] - limit_x[i])**2)
            prob_array[j] = 0.5*formula
        marg_posterior_x[i] = np.sum(prob_array)

    return marg_posterior_x


def marginal_y(data_array, x_0, y_0, shore_limit):

    limit_x = np.arange(0.5, x_0 + 50, 0.5)
    limit_y = np.arange(0.5, 10 + y_0, 0.5)
    prob_array = [0]*len(limit_x)
    marg_posterior_y = [0]*len(limit_y)

    for i in range(len(limit_y)):
        for j in range(len(limit_x)):
            formula = np.size(data_array) * np.log(limit_y[i]) - np.size(data_array)*np.log(np.pi)
            for n in range(len(data_array)):
                formula = formula - np.log(limit_y[i]**2 + (data_array[n] - limit_x[j])**2)
            prob_array[j] = 0.5*formula
        marg_posterior_y[i] = np.sum(prob_array)

    return marg_posterior_y
'''



'''
def credible_interval_finder(posterior_array):
    posterior_graph_area = np.sum(posterior_array)
    normalizing_const = 1/posterior_graph_area
    max_posterior = max(posterior_array)   # getting maximum posterior value
    start_index = posterior_array.index(max_posterior)   # start_index tells the element index of maximum posterior
    cred_interval = 0
    x_left = start_index
    left_line = start_index
    right_line = start_index
    while int(round(cred_interval*100)) < 96:
        if start_index == posterior_array.index(max_posterior):
             cred_interval = posterior_array[start_index]*normalizing_const
             x_left = start_index - 1
             start_index += 1
        elif x_left > 0 and start_index < len(posterior_array)-1:
             cred_interval = cred_interval + (posterior_array[x_left]+posterior_array[start_index]) * normalizing_const
             left_line = x_left
             right_line = start_index
             x_left = x_left - 1
             start_index += 1
        elif start_index == (len(posterior_array)-1):
             cred_interval = cred_interval + posterior_array[x_left]*normalizing_const
             right_line = start_index
             x_left = x_left - 1
        elif x_left == 0:
             cred_interval = cred_interval + posterior_array[start_index]*normalizing_const
             left_line = x_left
             start_index = start_index + 1


    return left_line, right_line
'''

def light_house(x_0, y_0, N, shore_limit):
    '''light_house-function input parameters : (x_0,y_0,N)'''
    angle_array = random_angle_generator(N)
    data_array = position_converter(angle_array, x_0, y_0)
    hist_range = [0, x_0 + 100]

    #posterior_array, limit = posterior(position_array, limit, x_0, y_0)

    #left_line, right_line = credible_interval_finder(posterior_array)

    #position_average = np.sum(data_array) / N

    X, Y, Z, limit_x, limit_y = posterior_2(data_array, x_0, y_0)

    marg_posterior_x = marginal_x(Z, limit_x, limit_y)

    marg_posterior_y = marginal_y(Z, limit_x, limit_y)

    '''making subplots'''

    '''Finding maximum value of posterior'''

    #posterior_max = np.argmax(posterior_array)  # returns the index for the maximum value in posterior_array
    #position_max = limit[posterior_max]  # finds the x-position for the best estimate

    plt.style.use('ggplot')

    plt.subplot(221)
    plt.hist(data_array, 'fd', hist_range, normed=False, weights=None, density=None)
    plt.title('Uniformly distributed data')
    plt.ylabel('count')
    #plt.legend(N, loc='upper right', shadow=True, prop={'size': 10})

    plt.subplot(224)
    plt.contour(X, Y, Z, 4, colors='black')
    plt.title('Prob(x,y|{Data},I) contour plot', fontsize=10)
    plt.xlabel('x')
    plt.ylabel('y')


    plt.subplot(222)
    plt.plot(limit_x, marg_posterior_x)
    #plt.vlines(position_average, -2000, 100, colors='g')
    #plt.hlines(min(posterior_array) - 100, left_line, right_line, colors='b')
    plt.title('P(x|I) vs. x', fontsize=10)
    plt.ylabel('prob(x|I)')
    plt.tick_params(axis='y')
    #max_posterior_x = max(marg_posterior_x)
    #max_index_x = marg_posterior_x.index(max_posterior_x)
    #plt.legend((limit_x[max_index_x]), loc='upper right', shadow=True, prop={'size': 10})

    plt.subplot(223)
    plt.plot(limit_y, marg_posterior_y)
    plt.ylabel('prob(y|I)')
    plt.xlabel('y')

    plt.show()


#######################################################################################################################
# light_house-function input parameters : (x_0, y_0, N, shore_limit)
# x_0,y_0= position of light house, N = number of data, shore_limit = distance of the shore that we want to study

light_house(100, 10, 90, 400)







