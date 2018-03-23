
import numpy as np
import matplotlib.pyplot as plt

# Generates a vector containing N random angles ranging from 0 to pi
def random_angle_generator(N):
    angle_array = np.random.uniform(0, np.pi, size=N)
    return angle_array



# converts the generated angles into x-positions
def position_converter(angle_array, x_0, y_0):
    position_array = np.zeros(len(angle_array))
    for i in range(len(angle_array)):
        if angle_array[i] == 0:
            angle_array[i] = 0.001
            position_array[i] = x_0 - y_0/np.tan(angle_array[i])

        elif angle_array[i] == 2*np.pi:
             angle_array[i] = np.pi - 0.001
             position_array = x_0 + abs(y_0/(np.tan(angle_array[i])-np.pi/2))

        elif angle_array[i] == np.pi/2:
            position_array[i] =x_0

        elif angle_array[i] > np.pi/2 and angle_array[i] != np.pi:
            position_array[i] = x_0 + y_0 / (np.tan(angle_array[i]-np.pi/2))

        elif angle_array[i] < np.pi/2 and angle_array[i] != 0:
            position_array[i] = x_0 - y_0/np.tan(angle_array[i])

    return position_array


def posterior(position_array,x_0, y_0):
    # using a flat prior, the x_0 input is not necessary since an arbitrary limit/range can be used
    limit = np.arange(0, 1500)
    log_array = np.zeros((np.size(limit), np.size(position_array)))
    posterior_array = [0]*len(limit)
    #posterior_array = np.zeros(np.size(limit))

    for i in range(np.size(limit)):
        for k in range(0, np.size(position_array)):
            log_array[i][k] = - np.log(np.power(y_0, 2) + np.power(position_array[k] - (limit[i]), 2))

    for i in range(np.size(limit)):
        '''posterior_array is a vector containing posterior for each integer alpha values'''
        posterior_array[i] = np.sum(log_array[i])  # sums up all the log terms for a fixed alpha

    return posterior_array, limit


'''The function below estimated posterior without taking the log, this will result in posterior --> 0 as the data set
    increases ( which is incorrect). 
    
    likelihood_matrix = np.zeros((np.size(limit), np.size(position_array)))
    posterior_array = np.zeros(len(limit))
    for i in range(np.size(limit)):
            for j in range(0,len(position_array)):
                likelihood_matrix[i][j] = y_0/(np.pi*(np.power(y_0, 2) + np.power((position_array[j] - limit[i]), 2)))
    for i in range(0, np.size(limit)):
        posterior_array[i] = np.prod(likelihood_matrix[i][:])'''


'''we will use a large enough data set so that the posterior will converge to one peak'''
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





def light_house(x_0,y_0,N):
       angle_array = random_angle_generator(N)
       position_array = position_converter(angle_array, x_0, y_0)
       hist_range=[0, 1500]

       posterior_array, limit = posterior(position_array, x_0, y_0)
       position_average = np.sum(position_array)/N

       left_line, right_line = credible_interval_finder(posterior_array)

       '''for testing if credible interval function seems realistic'''
       left_line_test, right_line_test = credible_interval_test(posterior_array)

       '''making subplots'''

       '''Finding maximum value of posterior'''

       posterior_max = np.argmax(posterior_array)    # returns the index for the maximum value in posterior_array
       position_max = limit[posterior_max]           # finds the x-position for the best estimate

       plt.style.use('ggplot')
       fig, axes = plt.subplots(2, sharex=True)
       ax1, ax2 = axes.ravel()
       ax1.hist(position_array, 'fd', hist_range, normed=False, weights=None, density=None)

       ax2.plot(limit, posterior_array)

       ax1.set_title('Light house: data histogram and posterior graph')
       ax1.set_xlabel('Positions along the shore')
       ax1.set_ylabel('Counts')

       ax2.set_xlabel('Positions along the shore')
       ax2.set_ylabel('Posterior graph')

       ax2.vlines(position_average, -2000, 100, colors='g')
       ax2.hlines(min(posterior_array)- 100, left_line, right_line, colors='b')
       ax2.hlines(min(posterior_array)-250,left_line_test, right_line_test, colors='r')

       ax1.legend((N, round(position_average)), loc='upper right', shadow=True)
       ax2.legend((position_max, (round(position_average)), 'CI=95%', '50%'), loc='upper right', shadow=True)

       fig.tight_layout()
       plt.show()



#######################################################################################################################
# light_house-function input parameters : (x_0,y_0,N)
# x_0,y_0= position of light house, N = number of data
# the coast line is 400 meter long, i.e the x-array or posterior_array has 400 elements

light_house(750,10,300)






