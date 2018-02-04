
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

        elif angle_array[i]== 2*np.pi:
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



def light_house(x_0,y_0,N):
       angle_array=random_angle_generator(N)
       position_array=position_converter(angle_array,x_0,y_0)

       hist_range=[x_0-500,x_0+500]
       plt.hist(position_array, 'fd', hist_range, normed=False, weights=None, density=None)  # determining bin size
                                                                                             # with Freedman-method
       plt.ylabel('Number of counts')
       plt.xlabel('Measurement variable')
       plt.title('Histogram of data points')
       plt.show()

       print('Data sample:', position_array)


#######################################################################################################################
# light_house-function input parameters : (x_0,y_0,N)
# x_0,y_0= position of light house, N = number of data

light_house(50,10,1000)





