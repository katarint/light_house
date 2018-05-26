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

    max_value = Z.max()
    for j in np.arange(1, np.size(y)):           # scaling down z
        for i in np.arange(1, np.size(x)):
            Z[j][i] = Z[j][i]/max_value
    print('z max : %s', Z.max())
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
    cred_array = [0.9, 0.7, 0.50, 0.3]   # array containing all credible percentages
    cred_index_x = np.zeros((4,), dtype=[('x', 'i4'), ('y', 'i4')])
    start_index_x = np.argmax(marg_posterior_x)

    brute_levels = np.zeros(len(cred_array))
    max_value = np.amax(marg_posterior_x)

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
        brute_levels[i] = marg_posterior_x[cred_index_x[i][0]]/max_value

    return cred_index_x, start_index_x, brute_levels



def cred_region_y(marg_posterior_y):
    cred_array = [0.9]
    cred_index_y = np.zeros((1,), dtype=[('x', 'i4'), ('y', 'i4')])   # a 4x2 zero matrix

    start_index_y = np.argmax(marg_posterior_y)  # start_index is the index of the element with highest value

    for i in range(len(cred_array)):
        count = np.amax(marg_posterior_y)*0.25  # the count starts with the initial start value
        L = start_index_y - 1
        R = start_index_y + 1
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




def m_h(data_array, num_walk):
    '''using MCMC with Metropolis algorithm'''
    x = 100
    y = 8
    mean = (x, y)
    cov = [[0.95, 0], [0, 1.3]]  # covariance matrix

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


def posterior(data_array, x, y):

    formula = 1

    for i in range(len(data_array)):
        formula = formula*(y/np.pi)*1/(y**2 + (data_array[i] - x)**2)

    return formula

def scatter_contour(x_burn_array, y_burn_array):
    x_contour_limit = np.arange(190, 212, 0.35)
    y_contour_limit = np.arange(3, 19, 0.35)
    z_contour = np.zeros((len(y_contour_limit), len(x_contour_limit)))
    X , Y = np.meshgrid(x_contour_limit, y_contour_limit)

    c = 0

    # binning data
    for iData in range(len(x_burn_array)):
        c = 0
        for i in range(len(y_contour_limit)):
            for j in range(len(x_contour_limit)):
                if x_burn_array[iData] < x_contour_limit[j] and y_burn_array[iData]< y_contour_limit[i]:
                   z_contour[i - 1][j - 1] += 1
                   c = 1
                if c == 1:
                    break
            if c == 1:
                break
    max_value = z_contour.max()

    for i in range(len(y_contour_limit)):
        for j in range(len(x_contour_limit)):
            z_contour[i][j] = z_contour[i][j]/max_value    # scaling the posterior
    return X, Y, z_contour, x_contour_limit, y_contour_limit


def marginal_x_scatter(z_contour): # calculates marginal posteriorer
    marg_scatter_x = np.sum(z_contour,0)
    norm_const = np.sum(marg_scatter_x)*0.35

    for i in range(len(marg_scatter_x)):
        marg_scatter_x[i] = marg_scatter_x[i]/norm_const

    return marg_scatter_x

def marginal_y_scatter(z_contour):  # calculates marginal posteriorer

    marg_scatter_y = np.sum(z_contour,1)
    norm_const = np.sum(marg_scatter_y)*0.35

    for i in range(len(marg_scatter_y)):
        marg_scatter_y[i] = marg_scatter_y[i]/norm_const


    return marg_scatter_y

def cred_mh_x(marg_scatter_x):
    cred_array = [0.9, 0.7, 0.5, 0.3]   # array containing all credible percentages
    cred_scatter_x = np.zeros((4,), dtype=[('x', 'i4'), ('y', 'i4')])
    start_index_scatter_x = np.argmax(marg_scatter_x)

    contour_levels = np.zeros(len(cred_array))   # values for contour levels
    max_value = np.amax(marg_scatter_x)


    for i in range(len(cred_array)):
        count = np.amax(marg_scatter_x)*0.35  # the count starts with the initial start value
        L = start_index_scatter_x - 1
        R = start_index_scatter_x + 1
        while count < cred_array[i]:
            if marg_scatter_x[L] >= marg_scatter_x[R]:
                count = count + marg_scatter_x[L]*0.35
                L = L - 1
            else:
                count = count + marg_scatter_x[R]*0.35
                R = R + 1
        cred_scatter_x[i][0] = L
        cred_scatter_x[i][1] = R
        contour_levels[i] = marg_scatter_x[cred_scatter_x[i][0]]/max_value


    return cred_scatter_x, start_index_scatter_x, contour_levels

def cred_mh_y(marg_scatter_y):
    cred_array = [0.9]   # array containing all credible percentages
    cred_scatter_y = np.zeros((1,), dtype=[('x', 'i4'), ('y', 'i4')])
    start_index_scatter_y = np.argmax(marg_scatter_y)

    for i in range(len(cred_array)):
        count = np.amax(marg_scatter_y)*0.35  # the count starts with the initial start value
        L = start_index_scatter_y - 1
        R = start_index_scatter_y + 1
        while count < cred_array[i]:

            if marg_scatter_y[L] >= marg_scatter_y[R]:
                count = count + marg_scatter_y[L]*0.35
                L = L - 1

            else:
                count = count + marg_scatter_y[R]*0.35
                R = R + 1
        cred_scatter_y[i][0] = L
        cred_scatter_y[i][1]= R

    return cred_scatter_y, start_index_scatter_y

def light_house(x_0, y_0, N, shore_limit):
    '''light_house-function input parameters : (x_0,y_0,N)'''
    angle_array = random_angle_generator(N)
    data_array = position_converter(angle_array, x_0, y_0)

    num_walk = 200000  # num_walk = number of random walks
    x_array, y_array, x_burn_array, y_burn_array, z_burn_array = m_h(data_array, num_walk)

    start_limit_x = x_0 - 20
    start_limit_y = y_0 - 5
    interval = 0.25

    #limit_x = np.arange(start_limit_x , x_0 + 20, interval)
    #limit_y = np.arange(start_limit_y, y_0 + 5, interval)



    #X_scatter, Y_scatter = np.meshgrid(limit_x, limit_y)


    #marginal_x = marg_x(limit_x, x_burn_array, y_burn_array)
    #marginal_y = marg_y(limit_y, x_burn_array, y_burn_array)

    #cred_index_x, start_index_x = cred_region_x(marginal_x)
    #cred_index_y, start_index_y = cred_region_y(marginal_y)

    X_scatter, Y_scatter, z_contour, x_contour_limit, y_contour_limit = scatter_contour(x_burn_array, y_burn_array)

    marg_scatter_x = marginal_x_scatter(z_contour)
    marg_scatter_y = marginal_y_scatter(z_contour)

    cred_scatter_x, start_index_scatter_x, contour_levels = cred_mh_x(marg_scatter_x)
    cred_scatter_y, start_index_scatter_y = cred_mh_y(marg_scatter_y)

    '''brute-force method'''
    X, Y, Z, limit_x, limit_y = posterior_2(data_array, x_0, y_0)

    marg_posterior_x = marginal_x(Z, limit_x, limit_y)

    marg_posterior_y = marginal_y(Z, limit_x, limit_y)

    cred_index_x, start_index_x, brute_levels = cred_region_x(marg_posterior_x) # contour_levels contains all levels for contour plot
    cred_index_y, start_index_y = cred_region_y(marg_posterior_y)

    '''making subplots'''
    ninety = 'darkred'
    seventy = 'orangered'
    fifty = 'indigo'
    thirty = 'lightseagreen'


    plt.figure()

    plt.style.use('ggplot')
    plt.subplot(121)
    plt.plot(x_array, y_array, '.', color='orchid', markersize=1)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('Metropolis-Hastings med 90000 stickprov', fontsize=12)
    plt.axis('equal')



    plt.subplot(122)
    plt.plot(x_burn_array, y_burn_array, '.', color='orchid', markersize=1)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('Med 1500 stickprov borträknade', fontsize=12)
    plt.axis('equal')

    plt.subplots_adjust(hspace=0.4)

    '''figure 2'''
    plt.figure()
    ax_2 = plt.subplot(221)
    plt.plot(x_contour_limit, marg_scatter_x, color='k')
    plt.plot(x_contour_limit, marg_scatter_x, '.', color='k')
    ax_2.set_xlim(x_0 - 10, x_0 + 10)

    plt.axvline(x=x_contour_limit[cred_scatter_x[3][0]], linestyle='--', color=thirty)     # 0,3 credible interval for x
    plt.axvline(x=x_contour_limit[cred_scatter_x[3][1]], linestyle='--', color=thirty)

    plt.axvline(x=x_contour_limit[cred_scatter_x[2][0]], linestyle='--', color=fifty)     # 0,5 credible interval for x
    plt.axvline(x=x_contour_limit[cred_scatter_x[2][1]], linestyle='--', color=fifty)


    plt.axvline(x=x_contour_limit[cred_scatter_x[1][0]], linestyle='--', color=seventy)     # 0,7 credible interval for x
    plt.axvline(x=x_contour_limit[cred_scatter_x[1][1]], linestyle='--', color=seventy)

    plt.axvline(x=x_contour_limit[cred_scatter_x[0][0]], linestyle='--', color=ninety)     # 0,9 credible interval for x
    plt.axvline(x=x_contour_limit[cred_scatter_x[0][1]], linestyle='--', color=ninety)

    # calculates x-intervals for 90% credibility and plots it as axis title for x
    cred_scatter_plus_x = x_contour_limit[cred_scatter_x[0][1]] - x_contour_limit[np.argmax(marg_scatter_x)]
    cred_scatter_minus_x = x_contour_limit[np.argmax(marg_scatter_x)] - x_contour_limit[cred_scatter_x[0][0]]
    ax_2.set_title(
        'position(x)= %s $\pm ^{%s} _{%s}$' % (np.round(x_contour_limit[np.argmax(marg_scatter_x)],2), np.round(cred_scatter_plus_x,2), np.round(cred_scatter_minus_x,2)))

    green_patch = mpatches.Patch(color=fifty, label='50%', ls='--')
    black_patch = mpatches.Patch(color=seventy, label='70%', ls='--')
    red_patch = mpatches.Patch(color=ninety, label='90%', ls='--')
    blue_patch = mpatches.Patch(color=thirty, label='30%', ls='--')

    plt.legend(handles=[blue_patch, green_patch, black_patch, red_patch], fontsize=15)

    ax1=plt.subplot(223)
    plt.contour(X_scatter, Y_scatter, z_contour, levels=[contour_levels[0], contour_levels[1], contour_levels[2], contour_levels[3]], colors='k', zorder = 2)
    plt.plot(x_burn_array, y_burn_array, '.', color='orchid', markersize=1, zorder = 1)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.axvline(x=x_contour_limit[cred_scatter_x[3][0]], linestyle='--', color=thirty)     # 0,3 credible interval for x
    plt.axvline(x=x_contour_limit[cred_scatter_x[3][1]], linestyle='--', color=thirty)

    plt.axvline(x=x_contour_limit[cred_scatter_x[2][0]], linestyle='--', color=fifty)     # 0,5 credible interval for x
    plt.axvline(x=x_contour_limit[cred_scatter_x[2][1]], linestyle='--', color=fifty)


    plt.axvline(x=x_contour_limit[cred_scatter_x[1][0]], linestyle='--', color=seventy)     # 0,7 credible interval for x
    plt.axvline(x=x_contour_limit[cred_scatter_x[1][1]], linestyle='--', color=seventy)

    plt.axvline(x=x_contour_limit[cred_scatter_x[0][0]], linestyle='--', color=ninety)     # 0,9 credible interval for x
    plt.axvline(x=x_contour_limit[cred_scatter_x[0][1]], linestyle='--', color=ninety)

    plt.axhline(y=y_contour_limit[cred_scatter_y[0][1]], linestyle='--', color=ninety)
    plt.axhline(y=y_contour_limit[cred_scatter_y[0][0]], linestyle='--', color=ninety)


    ax1.set_xlim(x_0 - 10, x_0 + 10)
    ax1.set_ylim(3, y_0 + 10)

    ax_3 = plt.subplot(224)
    plt.plot(marg_scatter_y, y_contour_limit, color='k')
    plt.plot(marg_scatter_y, y_contour_limit, '.', color='k')
    ax_3.set_ylim(3, y_0 + 10)
    plt.axhline(y=y_contour_limit[cred_scatter_y[0][1]], linestyle='--', color=ninety)
    plt.axhline(y=y_contour_limit[cred_scatter_y[0][0]], linestyle='--', color=ninety)

    # calculates x-intervals for 90% credibility and plots it as axis title for x
    cred_scatter_plus_y = y_contour_limit[cred_scatter_y[0][1]] - y_contour_limit[np.argmax(marg_scatter_y)]
    cred_scatter_minus_y = y_contour_limit[np.argmax(marg_scatter_y)] - y_contour_limit[cred_scatter_y[0][0]]
    ax_3.set_title(
        'position(x)= %s $\pm ^{%s} _{%s}$' % (
        np.round(y_contour_limit[np.argmax(marg_scatter_y)], 2), np.round(cred_scatter_plus_y, 2), np.round(cred_scatter_minus_y, 2)))

    '''figure 3'''
    plt.figure()

    plt.style.use('ggplot')

    ax1 = plt.subplot(223)
    plt.contour(X, Y, Z, levels=brute_levels, colors='k')
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tick_params(axis='y', which='major', labelsize=14)
    ax1.set_xlim(x_0 - 10, x_0 + 10)
    ax1.set_ylim(0, y_0 + 15)
    plt.axvline(x=limit_x[cred_index_x[3][0]], linestyle='--', color=thirty)  # 0,3 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[3][1]], linestyle='--', color=thirty)

    plt.axvline(x=limit_x[cred_index_x[0][0]], linestyle='--', color=ninety)  # 0,9 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[0][1]], linestyle='--', color=ninety)

    plt.axvline(x=limit_x[cred_index_x[1][0]], linestyle='--', color=seventy)  # 0,7 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[1][1]], linestyle='--', color=seventy)

    plt.axvline(x=limit_x[cred_index_x[2][0]], linestyle='--', color=fifty)  # 0,5 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[2][1]], linestyle='--', color=fifty)

    plt.axhline(y=limit_y[cred_index_y[0][0]], linestyle='--', color=ninety)  # 0,9 credible interval for y
    plt.axhline(y=limit_y[cred_index_y[0][1]], linestyle='--', color=ninety)


    ax2 = plt.subplot(221)
    plt.plot(limit_x, marg_posterior_x, color='black')
    plt.title('Marginal distribution for x', fontsize=15)

    # calculates x-intervals for 90% credibility and plots it as axis title for x
    cred_plus_x = limit_x[cred_index_x[0][1]] - limit_x[np.argmax(marg_posterior_x)]
    cred_minus_x = limit_x[np.argmax(marg_posterior_x)] - limit_x[cred_index_x[0][0]]
    ax2.set_title(
        'position(x)= %s $\pm ^{%s} _{%s}$' % (limit_x[np.argmax(marg_posterior_x)], cred_plus_x, cred_minus_x))

    plt.ylabel('p(x|D,I)', fontsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tick_params(axis='y', which='major', labelsize=14)
    ax2.set_xlim(x_0 - 10, x_0 + 10)

    plt.axvline(x=limit_x[cred_index_x[3][0]], linestyle='--', color=thirty)  # 0,3 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[3][1]], linestyle='--', color=thirty)

    plt.axvline(x=limit_x[cred_index_x[0][0]], linestyle='--', color=ninety)  # 0,9 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[0][1]], linestyle='--', color=ninety)

    plt.axvline(x=limit_x[cred_index_x[1][0]], linestyle='--', color=seventy)  # 0,7 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[1][1]], linestyle='--', color=seventy)

    plt.axvline(x=limit_x[cred_index_x[2][0]], linestyle='--', color=fifty)  # 0,5 credible interval for x
    plt.axvline(x=limit_x[cred_index_x[2][1]], linestyle='--', color=fifty)

    green_patch = mpatches.Patch(color=fifty, label='50%', ls='--')
    black_patch = mpatches.Patch(color=seventy, label='70%', ls='--')
    red_patch = mpatches.Patch(color=ninety, label='90%', ls='--')
    blue_patch = mpatches.Patch(color=thirty, label='30%', ls='--')

    plt.legend(handles=[blue_patch, green_patch, black_patch, red_patch], fontsize=15)

    plt.plot(limit_x, marg_posterior_x, '.', color='black')

    ax3 = plt.subplot(224)
    plt.plot(marg_posterior_y, limit_y, color='black')
    plt.xlabel('p(y|D,I)', fontsize=14)
    plt.title('Marginal distribution for y', fontsize=15)

    cred_plus_y = limit_y[cred_index_y[0][1]] - limit_y[np.argmax(marg_posterior_y)]
    cred_minus_y = limit_y[np.argmax(marg_posterior_y)] - limit_y[cred_index_y[0][0]]
    ax3.set_title(
        'position(y)= %s $\pm ^{%s} _{%s}$' % (limit_y[np.argmax(marg_posterior_y)], cred_plus_y, cred_minus_y))

    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tick_params(axis='y', which='major', labelsize=14)
    ax3.set_ylim(0, y_0 + 15)

    plt.axhline(y=limit_y[cred_index_y[0][0]], linestyle='--', color=ninety)
    plt.axhline(y=limit_y[cred_index_y[0][1]], linestyle='--', color=ninety)
    plt.plot(marg_posterior_y, limit_y, '.', color='black')

    plt.legend(handles=[red_patch], fontsize=15)

    plt.show()






#######################################################################################################################
# light_house-function input parameters : (x_0, y_0, N, shore_limit)
# x_0,y_0= position of light house, N = number of data, shore_limit = distance of the shore that we want to study

light_house(200, 10, 92, 400)







