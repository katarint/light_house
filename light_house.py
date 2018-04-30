
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def random_angle_generator(N):
    '''Generates a vector containing N random angles ranging from 0 to pi'''
    angle_array = np.random.uniform(0, np.pi, size=N)
    return angle_array




def position_converter(angle_array, x_0, y_0):
    '''converts the generated angles into x-positions (or data values)'''
    data_array = np.zeros(len(angle_array))
    for i in range(len(angle_array)):
        if angle_array[i] == 0:
            angle_array[i] = 0.001
            data_array[i] = x_0 - y_0/np.tan(angle_array[i])

        elif angle_array[i] == 2*np.pi:
             angle_array[i] = np.pi - 0.001
             data_array = x_0 + abs(y_0/(np.tan(angle_array[i])-np.pi/2))

        elif angle_array[i] == np.pi/2:
            data_array[i] =x_0

        elif angle_array[i] > np.pi/2 and angle_array[i] != np.pi:
            data_array[i] = x_0 + y_0 / (np.tan(angle_array[i]-np.pi/2))

        elif angle_array[i] < np.pi/2 and angle_array[i] != 0:
            data_array[i] = x_0 - y_0/np.tan(angle_array[i])

    return data_array

def mean_value(data_array):

    mean = int(round(np.sum(data_array) / np.size(data_array)))
    return mean



def posterior(data_array, x_0, y_0, prior, mean):
    '''creates a vector containing all the posteriors for every position x. The posterior is scaled for a nicer plot.'''


    x_range = np.arange(x_0-200 , x_0+200, 0.1)   # x ranges from x_0 -150 to x_0 + 150, with dx = 0.1

    posterior_array = [0, ]*np.size(x_range)
    log_array, best_estimate = log_posterior(data_array, y_0, x_range, prior)   # calling the function log_posterior

    normal_const = normalization(log_array, x_range)     # the function normalization calculates the global likelihood
    for i in range(len(x_range)):
        posterior_array[i] = np.exp(log_array[i])*normal_const    # taking the exponential of every elements in log_array

    print(np.sum(posterior_array)*0.1)   # to check if normal_const was calculated correctly

    return posterior_array, x_range, best_estimate



def log_posterior(data_array, y_0, x_range, prior):
    '''log_posterior takes the logarithm of posterior and scales it. Returning a vector log_array'''

    log_array = [0, ]*np.size(x_range)

    for i in range(np.size(x_range)):
        formula = 0
        for k in range(len(data_array)):
            formula = formula - np.log(y_0**2 + (data_array[k] - x_range[i])**2)
        log_array[i] = formula + np.log(prior)

    max_log_array = max(log_array)
    best_estimate_index = log_array.index(max_log_array)
    best_estimate = int(round(x_range[best_estimate_index]))

    for i in range(len(log_array)):
        log_array[i] = log_array[i] - max_log_array

    return log_array, best_estimate

def normalization(log_array, x_range):
    '''calculates the normalization constant (= global likelihood) for the posterior pdf'''

    temporary = [0, ]*np.size(x_range)
    for i in range(len(x_range)):
        temporary[i] = np.exp(log_array[i])     # taking the exponential of every elements in log_array

    normal_const = 1/(np.sum(temporary)*0.1)

    return normal_const



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

def credible_interval_test(posterior_array):
    posterior_graph_area = np.sum(posterior_array)
    normalizing_const = 1/posterior_graph_area
    max_posterior = max(posterior_array)   # getting maximum posterior value
    start_index = posterior_array.index(max_posterior)   # start_index tells the element index of maximum posterior
    cred_interval = 0
    x_left = 0
    left_line_test = 0
    right_line_test = 0
    while int(round(cred_interval*100)) < 50.5:
        cred_interval = cred_interval + posterior_array[right_line_test]*normalizing_const
        right_line_test += 1

    return left_line_test, right_line_test





def light_house(x_0, y_0, N, limit):
       angle_array = random_angle_generator(N)
       data_array = position_converter(angle_array, x_0, y_0)
       hist_range=[x_0 -150, x_0 + 150]
       mean = mean_value(data_array)

       prior = 1/limit   # choosing a uniform (flat) prior

       posterior_array, x_range, best_estimate = posterior(data_array, x_0, y_0, prior, mean)

       '''making subplots'''


       posterior_max = np.argmax(posterior_array)    # returns the index for the maximum value in posterior_array

       plt.style.use('ggplot')

       plt.style.use('ggplot')

       plt.subplot(121)
       plt.hist(data_array, 'fd', hist_range, normed=False, weights=None, density=None)
       plt.xlabel('Positions along the shore', fontsize=15)
       plt.ylabel('Counts', fontsize=15)
       plt.title('Light house data histogram', fontsize=15)
       plt.legend((N, round(mean)), loc='upper right', shadow=True, fontsize=15)
       plt.tick_params(axis='x', which='major', labelsize=14)
       plt.tick_params(axis='y', which='major', labelsize=14)


       ax2 = plt.subplot(122)
       plt.plot(x_range, posterior_array)
       plt.xlabel('Positions along the shore', fontsize=15)
       plt.ylabel('P(x|y,I)', fontsize=15)
       plt.title('Light house posterior pdf', fontsize=15)
       plt.vlines(mean, 0.01, 0.4, colors='g')
       red_patch = mpatches.Patch(color='orangered', label=best_estimate)
       green_patch = mpatches.Patch(color='g', label=mean)
       plt.legend(handles=[red_patch, green_patch], fontsize=15)
       plt.tick_params(axis='x', which='major', labelsize=14)
       plt.tick_params(axis='y', which='major', labelsize=14)
       ax2.set_xlim(x_0-200, x_0 +200)

       plt.show()


#######################################################################################################################
# light_house-function input parameters : (x_0, y_0, N, limit)
# x_0,y_0= position of light house, N = number of data
# the coast line is 400 meter long (limit = 400)

light_house(200, 10, 500, 400)






