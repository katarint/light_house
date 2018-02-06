
import numpy as np
import matplotlib.pyplot as plt

# Generates a vector containing random angles ranging from 0 to pi
def random_angle_generator(N):
    angle_array = np.random.uniform(0, np.pi, size=N)
    return angle_array


# converts the generated angles into x-positions
def position_converter(angle_array,x_0,y_0):
    position_array = np.zeros(len(angle_array))
    for i in range(len(angle_array)):
        if angle_array[i] == 0:
            angle_array[i] = 0.001
            position_array[i] = x_0 - y_0/np.tan(angle_array[i])

        elif angle_array[i] == 2*np.pi:
             angle_array[i] = np.pi - 0.001
             position_array = x_0 + abs(y_0/np.tan(angle_array[i]))

        elif angle_array[i] == np.pi/2:
            angle_array[i] = np.pi/2 - 0.001
            position_array[i] =x_0 - y_0 / np.tan(angle_array[i])

        elif angle_array[i] > np.pi/2 and angle_array[i] != np.pi:
            position_array[i] = x_0 + abs(y_0 / np.tan(angle_array[i]))

        elif angle_array[i] < np.pi/2 and angle_array[i] != 0 :
            position_array[i] = x_0 - y_0/np.tan(angle_array[i])

    return position_array


def posterior(position_array, y_0):       # using a flat prior
    limit = np.arange(-500, 500)
    likelyhood_matrix = np.zeros((np.size(limit), np.size(position_array)))
    posterior_array = np.zeros(len(limit))
    for i in range(np.size(limit)):
            for j in range(0,len(position_array)):
                likelyhood_matrix[i][j] = y_0/(np.pi*(np.power(y_0, 2) + np.power((position_array[j] - limit[i]), 2)))
    for i in range(0, np.size(limit)):
        posterior_array[i] = np.prod(likelyhood_matrix[i][:])


    return posterior_array




def light_house(x_0,y_0,N):
       angle_array=random_angle_generator(N)
       position_array=position_converter(angle_array,x_0,y_0)
       limit = np.arange(-500, 500)

       hist_range=[x_0-500,x_0+500]
       plt.hist(position_array, 'fd', hist_range, normed=False, weights=None, density=None)  # determining bin size
                                                                                             # with Freedman-method
       plt.ylabel('Number of counts')
       plt.xlabel('Measurement variable')
       plt.title('Histogram of data points')
       plt.show()
       posterior_array=posterior(position_array, y_0)
       # print('Data sample:', position_array)
       plt.plot(limit, posterior_array)
       plt.ylabel('Likelyhood(x)')
       plt.xlabel('light house position')
       plt.title('pdf-function')
       plt.show()





#######################################################################################################################
# light_house-function input parameters : (x_0,y_0,N)
# x_0,y_0= position of light house, N = number of data

light_house(50,10,100)







