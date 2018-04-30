import numpy as np
import matplotlib.pyplot as plt
import corner
import matplotlib.patches as mpatches


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
    '''using MCMC with Metropolis algorithm'''
    x = 100
    y = 8
    mean = (x, y)
    cov = [[0.95, 0], [0, 1.5]]  # covariance matrix

    x_array = [0]*num_walk
    y_array = [0]*num_walk
    z_array = [0]*num_walk

    x_burn_array = [0] * (num_walk-1500)
    y_burn_array = [0] * (num_walk-1500)
    z_burn_array = [0] * (num_walk-1500)

    for i in range(len(x_array)):
        u = np.random.random_sample()   # u is taken randomly from a normal distribution
        p_i = posterior(data_array, x, y)
        x_i, y_i = np.random.multivariate_normal(mean, cov).T
        p_f = posterior(data_array, x_i, y_i)
        r = p_f / p_i
        if u <= r:
            x_array[i] = x_i
            y_array[i] = y_i
            z_array[i] = p_f
            x = x_i
            y = y_i
            mean = (x_i, y_i)
        else:
            x_array[i] = x
            y_array[i] = y
            z_array[i] = p_i

    for i in range(len(x_burn_array)):
            x_burn_array[i] = x_array[i + 1500]
            y_burn_array[i] = y_array[i + 1500]
            z_burn_array[i] = z_array[i + 1500]

    return x_array, y_array, x_burn_array, y_burn_array, z_burn_array

def corner_plot(x_burn_array, y_burn_array):
    '''making corner plots and contour plot of the results from Metropolis Hasting samplings'''

    corner_array = np.zeros((len(x_burn_array), 2))
    for i in range(len(x_burn_array)):
        corner_array[i] = (x_burn_array[i], y_burn_array[i])

    return corner_array


def marg_x(limit_x, x_burn_array, y_burn_array):

    marginal_x = np.zeros(np.size(limit_x))

    for i in range(len(limit_x)-1):
        for j in range(len(x_burn_array)):
            if x_burn_array[j] <= limit_x[i+1] and x_burn_array[j] > limit_x[i]:
                marginal_x[i] = marginal_x[i] + y_burn_array[j]*0.25

    norm_x = norm_const_x(marginal_x)

    for i in range(len(marginal_x)):
        marginal_x[i] = marginal_x[i]*norm_x

    return marginal_x

def marg_y(limit_y, x_burn_array, y_burn_array):

    marginal_y = np.zeros(np.size(limit_y))
    print(np.size(x_burn_array))
    print(np.size(y_burn_array))

    for i in range(len(limit_y)-1):
        for j in range(len(y_burn_array)):
            if y_burn_array[j] <= limit_y[i+1] and y_burn_array[j] > limit_y[i]:
                marginal_y[i] = marginal_y[i] + x_burn_array[j]*0.25

    norm_y = norm_const_y(marginal_y)

    for i in range(len(marginal_y)):
        marginal_y[i] = marginal_y[i]*norm_y

    return marginal_y


def norm_const_x(marginal_x):
    norm_x = 1/(np.sum(marginal_x*0.25))
    return norm_x

def norm_const_y(marginal_y):
    norm_y = 1/(np.sum(marginal_y*0.25))
    return norm_y


def cred_region_x(marginal_x):
    cred_array = [0.95]
    cred_index_x = [0, ]*np.size(cred_array)
    start_index_x = np.argmax(marginal_x)

    for i in range(len(cred_array)):
        count = np.amax(marginal_x)*0.25  # the count starts with the initial start value
        k = 0
        while count < cred_array[i]:
              k += 1
              count = count + marginal_x[start_index_x+k]*0.25 + marginal_x[start_index_x-k]*0.25
        cred_index_x[i] = k

    return cred_index_x, start_index_x

def cred_region_y(marginal_y):
    cred_array = [0.95]
    cred_index_y = [0, ]*np.size(cred_array)
    start_index_y = np.argmax(marginal_y)

    for i in range(len(cred_array)):
        count = np.amax(marginal_y)*0.25  # the count starts with the initial start value
        k = 0
        while count < cred_array[i]:
              k += 1
              count = count + marginal_y[start_index_y+k]*0.25 + marginal_y[start_index_y-k]*0.25
        cred_index_y[i] = k

    return cred_index_y, start_index_y



def light_house(x_0, y_0, N, shore_limit):
    '''light_house-function input parameters : (x_0,y_0,N)'''
    angle_array = random_angle_generator(N)
    data_array = position_converter(angle_array, x_0, y_0)

    num_walk = 9000  # num_walk = number of random walks
    x_array, y_array, x_burn_array, y_burn_array, z_burn_array = m_h(data_array, num_walk)

    start_limit_x = x_0 - 20
    start_limit_y = y_0 - 5
    interval = 0.25

    limit_x = np.arange(start_limit_x , x_0 + 20, interval)
    limit_y = np.arange(start_limit_y, y_0 + 5, interval)

    X, Y = np.meshgrid(limit_x, limit_y)


    marginal_x = marg_x(limit_x, x_burn_array, y_burn_array)
    marginal_y = marg_y(limit_y, x_burn_array, y_burn_array)

    corner_array = corner_plot(x_burn_array, y_burn_array)

    cred_index_x, start_index_x = cred_region_x(marginal_x)
    cred_index_y, start_index_y = cred_region_y(marginal_y)


    '''making subplots'''

    plt.style.use('ggplot')

    plt.subplot(221)
    plt.plot(x_array, y_array, '.', color='m', markersize=1)
    #plt.contour(x_array, y_array)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('MCMC Metropolis Hastings, 9000 random walks', fontsize=10)
    plt.axis('equal')


    ax2=plt.subplot(224)
    plt.plot(limit_y, marginal_y)
    plt.xlabel('y')
    plt.title('Marginal distribution for y', fontsize=9)
    y_patch = mpatches.Patch(color='green', label=limit_y[np.argmax(marginal_y)])
    plt.legend(handles=[y_patch])
    plt.axvline(x=limit_y[np.argmax(marginal_y)], ls='--', color='green')
    plt.axvline(x=limit_y[start_index_y + cred_index_y], linestyle='--', color='k')
    plt.axvline(x=limit_y[start_index_y - cred_index_y], linestyle='--', color='k')
    cred_plus_y = limit_y[start_index_y + cred_index_y[0]]
    cred_minus_y = limit_y[start_index_y - cred_index_y[0]]
    ax2.set_title('position(y)= %s $\pm _{%s} ^{%s}$' % (limit_y[np.argmax(marginal_y)], cred_plus_y, cred_minus_y))

    ax1=plt.subplot(223)
    plt.plot(limit_x, marginal_x)
    plt.xlabel('x')
    plt.title('Marginal posterior for x', fontsize=9)
    x_patch = mpatches.Patch(color='green', label=limit_x[np.argmax(marginal_x)])
    plt.legend(handles=[x_patch])
    plt.axvline(x=limit_x[np.argmax(marginal_x)], ls='--', color='green')
    plt.axvline(x=limit_x[start_index_x + cred_index_x], linestyle='--', color='k')
    plt.axvline(x=limit_x[start_index_x - cred_index_x], linestyle='--', color='k')
    cred_plus_x = limit_x[start_index_x + cred_index_x[0]]
    cred_minus_x = limit_x[start_index_x - cred_index_x[0]]
    ax1.set_title('position(x)= %s $\pm _{%s} ^{%s}$' % (limit_x[np.argmax(marginal_x)], cred_plus_x, cred_minus_x))


    plt.subplot(222)
    plt.plot(x_burn_array, y_burn_array, '.', color='m', markersize=1)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('MCMC with 1500 steps as burn-in period', fontsize=10)
    plt.axis('equal')

    plt.subplots_adjust(hspace=0.4)


    figure = corner.corner(corner_array, bins = 20, quantiles=(0.16, 0.5, 0.84), show_titles=True, labels=[r"$position(x)$", r"$position(y)$"], fontsize=15)


    plt.show()









#######################################################################################################################
# light_house-function input parameters : (x_0, y_0, N, shore_limit)
# x_0,y_0= position of light house, N = number of data, shore_limit = distance of the shore that we want to study

light_house(200, 10, 93, 400)







