import numpy as np
import os
import device_module as dm
import matplotlib.pyplot as plt

# This code applies the virtual nodes method to build the reservoirs

# RC parameters
positive_voltage_stm = 0.95
negative_voltage_stm = -0.95
positive_voltage_pc = positive_voltage_stm
negative_voltage_pc = negative_voltage_stm
Batch_size = 800  # batch size


# for classification
def waveform_generator(current_limited, number_wave):
    """
    a function used to create random wave(sine or square)
    :param current_limited: the range of voltages
    :param number_wave: number of waves
    :return: wave_points
    """
    # 8 points to express waveform
    random_pulse = np.random.randint(0, 2, int(number_wave))
    print('random:{}'.format(random_pulse))
    wave_points = []
    for i in random_pulse:
        if i == 0:
            # sine function
            sine_points = [0, current_limited / 2, current_limited, current_limited / 2, 0,
                           -current_limited / 2, -current_limited, -current_limited / 2]
            wave_points = wave_points + sine_points
        elif i == 1:
            # square function
            square_points = [current_limited] * 4 + [-current_limited] * 4
            wave_points = wave_points + square_points
        # elif ac_value == 2:
        #     # cos function
        #     cos_points = [0, -current_limited / 2, -current_limited, -current_limited / 2, 0,
        #                   current_limited / 2, current_limited, current_limited / 2]
        #     wave_points = wave_points + cos_points
    print('wave:{}'.format(wave_points))

    return wave_points, list(random_pulse)


# build a random network
def train_stm(a0, b0, c0, delay_time=0, nodes_number_stm=50, train_epoch=1, bias_stm=1):
    """
    :param a0: the initial state of magnetization vector at x axis
    :param b0: the initial state of magnetization vector at y axis
    :param c0: the initial state of magnetization vector at z axis
    :param delay_time: the delay time of the short term memory task
    :param nodes_number_stm: the size of reservoirs
    :param train_epoch: the number of training times, which should be equal 1
    :param bias_stm: Bias term of neuron network
    :return: not any return, because all of weights will be saved as .npy files
    """
    # find the last weight matrix data
    if os.path.exists(
            'weight_matrix_comp/weight_out_stm_{}.npy'.format(delay_time)):
        weight_out_stm = np.load(
            'weight_matrix_comp/weight_out_stm_{}.npy'.format(delay_time))
        print('\r' + 'Loading weight_stm_{} matrix of STM successfully !'.format(delay_time), end='',
              flush=True)

    else:
        weight_out_stm = np.random.randint(-1, 2, (1, nodes_number_stm + 1))
        print('\r weight matrix of STM created successfully !', end='', flush=True)

    print(f'\rSTM  Time:{delay_time}')
    print('----------------------------------------------------------------')
    print('start to train !', flush=True)

    # fabricate the input, target, and output signals
    for j in range(0, train_epoch):
        s_in_stm, train_signal = waveform_generator(positive_voltage_stm, int(Batch_size / 8))
        # s_in_stm = np.random.randint(0, 2, Batch_size)
        y_out_list = []
        x_final_matrix = []
        mx_large_matrix = []
        m_x0, m_y0, m_z0 = a0, b0, c0
        # create pulse list
        for i1 in range(0, len(s_in_stm)):
            # if s_in_stm[i1] == 1:
            #     current_density = positive_voltage_stm
            # else:
            #     current_density = negative_voltage_stm

            mx_list, m_x0, m_y0, m_z0, x_matrix1 = dm.module_evolution(10000, mx=m_x0, my=m_y0, mz=m_z0,
                                                                       current_density=s_in_stm[i1],
                                                                       sample_points=nodes_number_stm)

            mx_large_matrix = mx_large_matrix + mx_list
            if (i1 + 1) % 8 == 0 and i1 != 0:
                number_interval = int(len(mx_large_matrix) / nodes_number_stm)
                if number_interval < 1:
                    number_interval = 1
                x_matrix1 = np.array(mx_large_matrix[1: len(mx_large_matrix):number_interval])

                while len(x_matrix1) != nodes_number_stm:
                    if len(x_matrix1) > nodes_number_stm:
                        x_matrix1 = x_matrix1[:-1]
                    else:
                        x_matrix1 = np.append(x_matrix1, x_matrix1[-1])
                x_matrix1 = np.reshape(x_matrix1, (nodes_number_stm, 1))
                print(x_matrix1.T)

                x_matrix1 = np.append(x_matrix1, 1).reshape(-1, 1)
                y_out = np.dot(weight_out_stm, x_matrix1)
                y_out_list.append(y_out[0, 0])
                x_final_matrix.append(x_matrix1.T.tolist()[0])
                mx_large_matrix = []

        # update weight
        y_train_matrix = np.array(train_signal).reshape(1, len(train_signal))
        x_final_matrix = np.asmatrix(x_final_matrix).T
        weight_out_stm = np.dot(y_train_matrix, np.linalg.pinv(x_final_matrix))

        # save weight matrix as .npy files
        np.save('weight_matrix_comp/weight_out_stm_{}.npy'.format(delay_time), weight_out_stm)

        # calculate the error
        error_learning = np.var(np.array(train_signal) - np.array(y_out_list))
        print('\rEpoch:{}/{}  error:{}'.format(j + 1, train_epoch, error_learning))

        print('Trained successfully !')
        print('----------------------------------------------------------------')
    return 0


def test_stm(a0, b0, c0, delay_time=0, test_points=20, bias_stm=1, node_points=50):
    """
    the function is used to test the nonlinear ability of short term memory of MTJ Rc neuron network
    :param a0: initial value of magnetization vector at x axis
    :param b0: initial value of magnetization vector at y axis
    :param c0: initial value of magnetization vector at z axis
    :param delay_time: delay time of short term memory
    :param test_points: the length of test-input signals
    :param bias_stm: bias term
    :param node_points: size of reservoirs
    :return: cor_stm, the correlation of target and actual output
    """

    # loading weights
    if os.path.exists('weight_matrix_comp/weight_out_stm_{}.npy'.format(delay_time)):
        weight_out_stm = np.load('weight_matrix_comp/weight_out_stm_{}.npy'.format(delay_time))
        print('\rweight_out_stm_{}:{}'.format(delay_time, weight_out_stm), end='')
        print('\rstart to test !', end='')

    else:
        print('no valid weight matrix for STM task!')
        return 0

    # a list used to save the output results
    y_out_list = []
    # produce test data
    s_in_stm, train_signal = waveform_generator(positive_voltage_stm, test_points)
    # s_in_stm = np.random.randint(0, 2, test_points)
    m_x0, m_y0, m_z0 = a0, b0, c0
    mx_large_matrix = []

    # create pulse list
    for i1 in range(0, len(s_in_stm)):

        mx_list, m_x0, m_y0, m_z0, x_matrix1 = dm.module_evolution(10000, mx=m_x0, my=m_y0, mz=m_z0,
                                                                   current_density=s_in_stm[i1],
                                                                   sample_points=node_points)

        mx_large_matrix = mx_large_matrix + mx_list
        if (i1 + 1) % 8 == 0 and i1 != 0:
            number_interval = int(len(mx_large_matrix) / node_points)
            if number_interval < 1:
                number_interval = 1
            x_matrix1 = np.array(mx_large_matrix[1: len(mx_large_matrix):number_interval])

            while len(x_matrix1) != node_points:
                if len(x_matrix1) > node_points:
                    x_matrix1 = x_matrix1[:-1]
                else:
                    x_matrix1 = np.append(x_matrix1, x_matrix1[-1])
            x_matrix1 = np.reshape(x_matrix1, (node_points, 1))
            print(x_matrix1.T)

            x_matrix1 = np.append(x_matrix1, 1).reshape(-1, 1)
            y_out = np.dot(weight_out_stm, x_matrix1)
            y_out_list.append(np.abs(y_out[0, 0]))

    error_learning = np.var(np.array(train_signal[int(delay_time):]) - np.array(y_out_list[int(delay_time):]))
    print('error:{}'.format(error_learning))
    print('train_signal:{}'.format(train_signal))
    print('y_out_list:{}'.format(y_out_list))

    # FIGURE
    plt.figure('Test results')
    plt.plot(train_signal, c='blue', label='target')
    plt.plot(y_out_list, c='green', label='module')
    plt.ylabel('Index')
    plt.xlabel('Time')
    plt.legend()
    # plt.show()

    plt.figure('Comparison')
    plt.subplot(2, 1, 1)
    plt.plot(train_signal, c='blue', label='target & input')
    plt.legend()
    plt.ylabel('Index')
    plt.xlabel('Time')

    plt.subplot(2, 1, 2)
    plt.plot(y_out_list, c='red', label='module')
    plt.legend()
    plt.ylabel('Index')
    plt.xlabel('Time')
    plt.show()

    return 0


if __name__ == '__main__':
    m_x, m_y, m_z = 1, 0.01, 0.01
    # train_stm(a0=1, b0=0.01, c0=0.01, nodes_number_stm=200)
    # test_stm(a0=1, b0=0.01, c0=0.01, test_points=80, node_points=200)
    x1 = [1, 1, 0]
    y1 = [1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 0]
    t1 = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7]
    x2 = [1, 1, 0, 0]
    t2 = [1, 3, 5]

    plt.figure()
    plt.plot(t1, y1, c='red')
    plt.scatter(t2, x1, c='blue')
    plt.ylabel('current')
    plt.xlabel('Time')

    plt.show()
