import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as plb


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
    limit_x = np.arange(x_0 - 30, x_0 + 30, 0.25)
    limit_y = np.arange(0, 15 + y_0, 0.25)
    x = limit_x
    y = limit_y
    X, Y = np.meshgrid(x, y)
    Z = log_posterior_2(x, y, data_array)

    for j in np.arange(1, np.size(y)):
        for i in np.arange(1, np.size(x)):
            Z[j][i] = np.exp(Z[j][i])

    return X, Y, Z, limit_x, limit_y


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
    '''Calculates the marginal distribution of x'''

    marg_posterior_x = [0, ]*np.size(limit_x)

    for i in range(len(limit_x)):
        marg_posterior_x[i] = 0
        for j in range(len(limit_y)):
            marg_posterior_x[i] = marg_posterior_x[i] + Z[j][i]*0.25

    norm_x = norm_const_x(marg_posterior_x)
    for i in range(len(limit_x)):  # normalizing the posterior
        marg_posterior_x[i] = marg_posterior_x[i] * norm_x

    print(np.sum(marg_posterior_x)*0.25)

    return marg_posterior_x


def marginal_y(Z, limit_x, limit_y):
    '''Calculates the marginal distribution of y'''

    marg_posterior_y = [0, ]*np.size(limit_y)

    for i in range(len(limit_y)):
        marg_posterior_y[i] = 0
        for j in range(len(limit_x)):
            marg_posterior_y[i] = marg_posterior_y[i] + Z[i][j]*0.25

    norm_y = norm_const_y(marg_posterior_y)

    for i in range(len(limit_y)):     # normalizing the posterior
        marg_posterior_y[i] = marg_posterior_y[i]*norm_y

    print(np.sum(marg_posterior_y) * 0.25)

    return marg_posterior_y

def norm_const_y(marg_posterior_y):
    '''calculates normalization constant for marginal posterior for y'''
    norm_y = 1/(np.sum(marg_posterior_y)*0.25)
    return norm_y

def norm_const_x(marg_posterior_x):
    '''calculates normalization constant for marginal posterior for x'''
    norm_x = 1/(np.sum(marg_posterior_x)*0.25)
    return norm_x


def cred_region_x(marg_posterior_x):
    cred_array = [0.5, 0.7, 0.90, 0.3]   # array containing all credible percentages
    cred_index_x = np.zeros((5,), dtype=[('x', 'i4'), ('y', 'i4')])
    start_index_x = np.argmax(marg_posterior_x)

    for i in range(len(cred_array)):
        count = np.amax(marg_posterior_x)*0.25  # the count starts with the initial start value
        L = start_index_x - 1
        R = start_index_x + 1
        while count < cred_array[i]:
            if marg_posterior_x[L] >= marg_posterior_x[R]:
                count = count + marg_posterior_x[L]*0.25
                L = L - 1
            else:
                count = count + marg_posterior_x[L]*0.25
                R = R + 1
        cred_index_x[i][0] = L
        cred_index_x[i][1]= R

    return cred_index_x, start_index_x



def cred_region_y(marg_posterior_y):
    cred_array = [0.5, 0.7, 0.90, 0.3]
    cred_index_y = np.zeros((5,), dtype=[('x', 'i4'), ('y', 'i4')])   # a 4x2 zero matrix

    start_index_y = np.argmax(marg_posterior_y)  # start_index is the index of the element with highest value

    for i in range(len(cred_array)):
        count = np.amax(marg_posterior_y)*0.25  # the count starts with the initial start value
        L = start_index_y - 1
        R = start_index_y + 1
        print(i)
        while count < cred_array[i]:
            if marg_posterior_y[L] >= marg_posterior_y[R]:
                count = count + marg_posterior_y[L]*0.25
                L = L - 1
            else:
                count = count + marg_posterior_y[L]*0.25
                R = R + 1
        cred_index_y[i][0] = L
        cred_index_y[i][1] = R


    return cred_index_y, start_index_y


def light_house(x_0, y_0, N, shore_limit):
    '''light_house-function input parameters : (x_0,y_0,N)'''
    angle_array = random_angle_generator(N)
    data_array = position_converter(angle_array, x_0, y_0)
    hist_range = [50, x_0 + 150]

    #posterior_array, limit = posterior(position_array, limit, x_0, y_0)

    #left_line, right_line = credible_interval_finder(posterior_array)

    #position_average = np.sum(data_array) / N

    X, Y, Z, limit_x, limit_y = posterior_2(data_array, x_0, y_0)

    marg_posterior_x = marginal_x(Z, limit_x, limit_y)

    marg_posterior_y = marginal_y(Z, limit_x, limit_y)

    cred_index_x, start_index_x = cred_region_x(marg_posterior_x) # contour_levels contains all levels for contour plot
    cred_index_y, start_index_y = cred_region_y(marg_posterior_y)

    '''making subplots'''

    plt.style.use('ggplot')

    ax1 = plt.subplot(223)
    plt.contour(X, Y, Z, 4, colors='k')
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tick_params(axis='y', which='major', labelsize=14)
    ax1.set_xlim(x_0 - 10, x_0 + 10)
    ax1.set_ylim(0, y_0 + 15)
    plt.axvline(x=limit_x[cred_index_x[3][0]], linestyle=':', color='blue')     # 0,3 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[3][1]], linestyle=':', color='blue')

    plt.axvline(x=limit_x[cred_index_x[0][0]], linestyle=':', color='darkmagenta')   # 0,5 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[0][1]], linestyle=':', color='darkmagenta')

    plt.axvline(x=limit_x[cred_index_x[1][0]], linestyle=':', color='teal')        # 0,7 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[1][1]], linestyle=':', color='teal')

    plt.axvline(x=limit_x[cred_index_x[2][0]], linestyle=':', color='crimson')     # 0,9 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[2][1]], linestyle=':', color='crimson')

    plt.axhline(y=limit_y[cred_index_y[2][0]], linestyle=':', color='crimson')     # 0,9 credible interval for y
    plt.axhline(y=limit_y[cred_index_y[2][1]], linestyle=':', color='crimson')

    plt.suptitle('Marginal distribution for x and y', fontsize=14, fontweight='bold')

    ax2 = plt.subplot(221)
    plt.plot(limit_x, marg_posterior_x, color='black')
    plt.title('Marginal distribution for x', fontsize=15)

    # calculates x-intervals for 90% credibility and plots it as axis title for x
    cred_plus_x = limit_x[cred_index_x[2][1]] - limit_x[np.argmax(marg_posterior_x)]
    cred_minus_x = limit_x[np.argmax(marg_posterior_x)] - limit_x[cred_index_x[2][0]]
    ax2.set_title('position(x)= %s $\pm _{%s} ^{%s}$' % (limit_x[np.argmax(marg_posterior_x)], cred_plus_x, cred_minus_x))

    plt.ylabel('prob(x|D,I)', fontsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tick_params(axis='y', which='major', labelsize=14)
    ax2.set_xlim(x_0 - 10, x_0 + 10)

    plt.axvline(x=limit_x[cred_index_x[3][0]], linestyle=':', color='blue')     # 0,3 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[3][1]], linestyle=':', color='blue')


    plt.axvline(x=limit_x[cred_index_x[0][0]], linestyle=':', color='darkmagenta')  # 0,5 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[0][1]], linestyle=':', color='darkmagenta')

    plt.axvline(x=limit_x[cred_index_x[1][0]], linestyle=':', color='teal')  # 0,7 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[1][1]], linestyle=':', color='teal')

    plt.axvline(x=limit_x[cred_index_x[2][0]], linestyle=':', color='crimson')  # 0,9 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[2][1]], linestyle=':', color='crimson')

    green_patch = mpatches.Patch(color='darkmagenta', label='50%', ls=':')
    black_patch = mpatches.Patch(color='teal', label='70%', ls=':')
    red_patch = mpatches.Patch(color='crimson', label='90%', ls=':')
    blue_patch = mpatches.Patch(color='blue', label='30%', ls=':')

    plt.legend(handles=[blue_patch,green_patch, black_patch, red_patch], fontsize=15)

    plt.plot(limit_x, marg_posterior_x, '.', color='black')



    ax3 = plt.subplot(224)
    plt.plot(marg_posterior_y, limit_y, color='black')
    plt.xlabel('prob(y|D,I)', fontsize=14)
    plt.title('Marginal distribution for y', fontsize=15)

    cred_plus_y = limit_y[cred_index_y[2][1]] - limit_y[np.argmax(marg_posterior_y)]
    cred_minus_y = limit_y[np.argmax(marg_posterior_y)] - limit_y[cred_index_y[2][0]]
    ax3.set_title('position(y)= %s $\pm _{%s} ^{%s}$' % (limit_y[np.argmax(marg_posterior_y)], cred_plus_y, cred_minus_y))

    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tick_params(axis='y', which='major', labelsize=14)
    ax3.set_ylim(0, y_0 + 15)


    plt.axhline(y=limit_y[cred_index_y[2][0]],linestyle=':', color='crimson')
    plt.axhline(y=limit_y[cred_index_y[2][1]],linestyle=':', color='crimson')
    plt.plot(marg_posterior_y, limit_y, '.', color='black')

    plt.legend(handles=[red_patch], fontsize=15)




    plt.show()


#######################################################################################################################
# light_house-function input parameters : (x_0, y_0, N, shore_limit)
# x_0,y_0= position of light house, N = number of data, shore_limit = distance of the shore that we want to study

light_house(200, 10, 120, 400)







