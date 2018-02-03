
import numpy as np
import math
import matplotlib.pyplot as plt

# Generates a vector containing random angles ranging from 0 to pi
def random_angle_generator(N):
    angle_array = np.random.uniform(0, np.pi, size=N)
    return angle_array


# converts the generated angles into x-positions
def position_converter(angle_array,x_0,y_0):
    temp = 0

    for i in range(len(angle_array)):
        if angle_array[i-temp] == 0 or angle_array[i-temp] == np.pi:
            del angle_array[i-temp]
            temp += 1

    hypotenuse_array = np.zeros(len(angle_array))    #hypotenuse_array contains all the values of the hypotenuse of data
    abs_x_array = np.zeros(len(angle_array))
    position_array = np.zeros(len(angle_array))

    for i in range(len(angle_array)):
        if angle_array[i] <= np.pi/2:
            hypotenuse_array[i]=y_0/np.sin(angle_array[i])
            abs_x_array[i]=hypotenuse_array[i]*np.cos(angle_array[i])
            position_array[i]=x_0-abs_x_array[i]

        elif angle_array[i] > np.pi/2:
            angle_array[i] = np.pi-angle_array[i]
            hypotenuse_array[i] = y_0 / np.sin(angle_array[i])
            abs_x_array[i] = hypotenuse_array[i] * np.cos(angle_array[i])
            position_array[i] = x_0 + abs_x_array[i]

    return position_array



def light_house(x_0,y_0,N):
       angle_array=random_angle_generator(N)
       position_array=position_converter(angle_array,x_0,y_0)
       plt.hist(position_array, 'fd', range=None, normed=False, weights=None, density=None)  # determining bin size
                                                                                             # with Freedman-method
       plt.ylabel('Number of counts')
       plt.xlabel('Measurement variable')
       plt.title('Histogram of data points')
       plt.show()

#######################################################################################################################
# light_house-function input parameters : (x_0,y_0,N)
# x_0,y_0= position of light house, N = number of data

light_house(50,10,500)



