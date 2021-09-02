import numpy as np
import os
import matplotlib.pyplot as plt
import device_oscillator as do

# hyper parameters
Batch_size = 2000


def train_stm(a0, b0, c0, delay_time=1, nodes_number_stm=25, bias_stm=1):
    """
    a function used to achieve training of classification by using oscillation MTJ
    :param a0: initial state of magnetization vector at x axis
    :param b0: initial state of magnetization vector at y axis
    :param c0: initial state of magnetization vector at z axis
    :param delay_time: delay time
    :param nodes_number_stm: number of nodes, reservoir size
    :param bias_stm: bias term
    :return: no any return
    """
    # find the last weight matrix data
    if os.path.exists(
            'weight_matrix_os/weight_out_stm_{}.npy'.format(delay_time)):
        weight_out_stm = np.load(
            'weight_matrix_os/weight_out_stm_{}.npy'.format(delay_time))
        print('\r' + 'Loading weight_stm_{} matrix of STM successfully !'.format(delay_time), end='',
              flush=True)

    else:
        weight_out_stm = np.random.randint(-1, 2, (1, nodes_number_stm + 1))
        print('\r weight matrix of STM created successfully !', end='', flush=True)

    print(f'\rSTM  Time:{delay_time}')
    print('----------------------------------------------------------------')
    print('start to train !', flush=True)

    # fabricate the input, target, and output signals
    train_epoch = 1
    for j in range(0, train_epoch):
        s_in_stm = np.random.randint(0, 2, Batch_size)
        y_out_list = []
        x_final_matrix = []
        m_x0, m_y0, m_z0 = a0, b0, c0

        # create pulse list
        for i1 in range(0, Batch_size):
            if s_in_stm[i1] == 1:
                current = -0.836
                ac_current = 0.2
                f_ac = 8e11
            else:
                current = -0.836
                ac_current = 0.1
                f_ac = 8e11
            mx_list, t_list, vol1, envelope_list, time_env_list, [m_x0, m_y0, m_z0], _ = do.evolution_mag(m_x0, m_y0,
                                                                                                          m_z0, current,
                                                                                                          ac_current,
                                                                                                          f_ac=f_ac)
            # plt.figure()
            # # plt.plot(t_list, mx_list)
            # plt.plot(time_env_list, envelope_list)
            # # plt.scatter(time_env_list[10:], envelope_list[10:])
            # print(len(envelope_list))
            # plt.show()

            # sampling the nodes from resistances list
            number_interval = int(len(envelope_list) / nodes_number_stm)
            print(len(envelope_list))
            x_matrix1 = np.array(envelope_list[1: len(envelope_list):number_interval])

            while len(x_matrix1) != nodes_number_stm:
                if len(x_matrix1) > nodes_number_stm:
                    x_matrix1 = x_matrix1[:-1]
                else:
                    x_matrix1 = np.append(x_matrix1, x_matrix1[-1])
            x_matrix1 = np.reshape(x_matrix1, (nodes_number_stm, 1))

            print(x_matrix1.T)
            x_matrix1 = np.append(x_matrix1, bias_stm).reshape(-1, 1)
            y_out = np.dot(weight_out_stm, x_matrix1)
            y_out_list.append(y_out[0, 0])
            x_final_matrix.append(x_matrix1.T.tolist()[0])

            if i1 % 100 == 0:
                print('\r Process:{}/{}'.format(i1, Batch_size), end='', flush=True)

        # update weight
        y_train_whole = list(s_in_stm)[-int(delay_time):] + list(s_in_stm)[0:-int(delay_time)]
        if delay_time == 0:
            y_train_whole = s_in_stm

        y_train_matrix = np.array(y_train_whole).reshape(1, len(y_train_whole))
        x_final_matrix = np.asmatrix(x_final_matrix).T
        weight_out_stm = np.dot(y_train_matrix, np.linalg.pinv(x_final_matrix))

        # save weight matrix as .npy files
        np.save('weight_matrix_os/weight_out_stm_{}.npy'.format(delay_time), weight_out_stm)

        # calculate the error
        error_learning = np.var(
            np.array(y_train_whole) - np.array(y_out_list))
        print('\rEpoch:{}/{}  error:{}'.format(j + 1, train_epoch, error_learning))

        print('Trained successfully !')
        print('----------------------------------------------------------------')

    return 0


def test_stm(a0, b0, c0, delay_time=1, test_points=20, nodes_number_stm=25, bias_stm=1):
    """
    the function is used to test the nonlinear ability of short term memory of MTJ Rc neuron network
    :param a0: initial value of magnetization vector at x axis
    :param b0: initial value of magnetization vector at y axis
    :param c0: initial value of magnetization vector at z axis
    :param delay_time: delay time of short term memory
    :param test_points: the length of test-input signals
    :param bias_stm: bias term
    :return: cor_stm, the correlation of target and actual output
    """

    # loading weights
    if os.path.exists('weight_matrix_os/weight_out_stm_{}.npy'.format(delay_time)):
        weight_out_stm = np.load('weight_matrix_os/weight_out_stm_{}.npy'.format(delay_time))
        print('\rweight_out_stm_{}:{}'.format(delay_time, weight_out_stm), end='')
        print('\rstart to test !', end='')

    else:
        print('no valid weight matrix for STM task!')
        return 0

    # a list used to save the output results
    y_out_list = []
    # produce test data
    s_in_stm = np.random.randint(0, 2, test_points)
    m_x0, m_y0, m_z0 = a0, b0, c0

    # create pulse list
    for i1 in range(0, len(s_in_stm)):
        if s_in_stm[i1] == 1:
            current = -0.836
            ac_current = 0.2
            f_ac = 8e11
        else:
            current = -0.836
            ac_current = 0.1
            f_ac = 8e11
        mx_list, t_list, vol1, envelope_list, time_env_list, [m_x0, m_y0, m_z0], _ = do.evolution_mag(m_x0, m_y0, m_z0,
                                                                                                      current,
                                                                                                      ac_current,
                                                                                                      f_ac=f_ac)

        # plt.figure()
        # plt.plot(t_list, vol1)
        # plt.scatter(time_env_list, envelope_list, c='red')
        # plt.show()

        # sampling the nodes from resistances list
        number_interval = int(len(envelope_list) / nodes_number_stm)
        x_matrix1 = np.array(envelope_list[1: len(envelope_list):number_interval])

        while len(x_matrix1) != nodes_number_stm:
            if len(x_matrix1) > nodes_number_stm:
                x_matrix1 = x_matrix1[:-1]
            else:
                x_matrix1 = np.append(x_matrix1, x_matrix1[-1])
        x_matrix1 = np.reshape(x_matrix1, (nodes_number_stm, 1))

        x_matrix1 = np.append(x_matrix1, bias_stm).reshape(-1, 1)
        y_out = np.dot(weight_out_stm, x_matrix1)
        # print(x_matrix1)
        # print(mx_list)
        y_out_list.append(y_out[0, 0])

    # fabricate the target signals
    y_train_whole = list(s_in_stm)[-int(delay_time):] + list(s_in_stm)[0:-int(delay_time)]
    if delay_time == 0:
        y_train_whole = s_in_stm
    error_learning = np.var(np.array(y_train_whole[int(delay_time):]) - np.array(y_out_list[int(delay_time):]))
    print('error:{}'.format(error_learning))

    t_step = list(range(0, len(s_in_stm)))
    t_step = [i1 * 20e-9 for i1 in t_step]

    # FIGURE
    plt.figure('Train')
    plt.title(r'delay task')
    plt.subplot(3, 1, 1)
    plt.plot(t_step[10:], s_in_stm[10:])
    plt.xlabel('T_step')
    plt.ylabel('S_in')
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t_step[10:], y_train_whole[10:])
    plt.xlabel('T_step')
    plt.ylabel('y_train_STM')
    plt.grid()

    plt.subplot(3, 1, 3)
    print('y_out:{}'.format(y_out_list))
    plt.plot(t_step[10:], y_out_list[10:])
    plt.xlabel('T_step')
    plt.ylabel('y_out_STM')
    plt.show()

    # calculate the correlation
    cor_stm = pow(np.corrcoef(y_out_list[int(delay_time):], y_train_whole[int(delay_time):])[0, 1], 2)

    return cor_stm


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
            sine_points = [0, current_limited / 2, current_limited, current_limited/2, 0,
                           -current_limited/2, -current_limited, -current_limited/2]
            wave_points = wave_points + sine_points
        else:
            # square function
            square_points = [current_limited]*4 + [-current_limited]*4
            wave_points = wave_points + square_points
    print('wave:{}'.format(wave_points))

    train_signal = []
    for i in random_pulse:
        temp = [i]*8
        train_signal = train_signal + temp
    print('train_signal:{}'.format(train_signal))
    return wave_points, list(random_pulse)


def train_classification(a0, b0, c0, current_limited, number_wave, nodes_number):
    """
    a function used to train weight of classification task
    :param a0: initial value of magnetization at x axis
    :param b0: initial value of magnetization at y axis
    :param c0: initial value of magnetization at z axis
    :param current_limited: the boundary of applied current
    :param number_wave: number of input wave signals
    :param nodes_number: size of reservoir
    :return: no any return, all of results have achieved in process
    """
    # wave generators
    # wave_points = waveform_generator(current_limited, number_wave)
    # find the last weight matrix data
    if os.path.exists('weight_matrix_os/weight_out_classification.npy'):
        weight_out_stm = np.load('weight_matrix_os/weight_out_classification.npy')
        print('\r' + 'Loading weight_out_classification matrix successfully !', end='', flush=True)

    else:
        # think about bias term
        weight_out_stm = np.random.randint(-1, 2, (1, nodes_number + 1))
        print('\r weight matrix of STM created successfully !', end='', flush=True)

    print('\rClassification')
    print('----------------------------------------------------------------')
    print('start to train !', flush=True)

    # fabricate the input, target, and output signals
    train_epoch = 1
    for j in range(0, train_epoch):
        s_in, train_signal = waveform_generator(current_limited, number_wave)
        print('---------------------------------------------------------------')
        y_out_list = []
        x_final_matrix = []
        m_x0, m_y0, m_z0 = a0, b0, c0
        # used to get the figure
        envelope_matrix = []

        # create pulse list
        for i1 in range(len(s_in)):
            mx_list, _, vol1, envelope_list, time_env_list, [m_x0, m_y0, m_z0], _ = do.evolution_mag(m_x0, m_y0, m_z0,
                                                                                                     magnitude=s_in[i1])
            # mx_list2, _, vol2, envelope_list2, time_env_list2, [m_x0, m_y0, m_z0], _ = do.evolution_mag(m_x0, m_y0,
            #                                                                                             m_z0,
            #                                                                                             magnitude=-i1)
            # envelope_list = envelope_list + envelope_list2
            envelope_matrix = envelope_matrix + envelope_list
            # sampling the nodes from resistances list
            if (i1+1) % 8 == 0:
                number_interval = int(len(envelope_matrix) / nodes_number)
                print('length of wave form: {}'.format(len(envelope_matrix)))
                if number_interval < 1:
                    number_interval = 1
                x_matrix1 = np.array(envelope_matrix[1: len(envelope_matrix):number_interval])

                while len(x_matrix1) != nodes_number:
                    if len(x_matrix1) > nodes_number:
                        x_matrix1 = x_matrix1[:-1]
                    else:
                        x_matrix1 = np.append(x_matrix1, x_matrix1[-1])
                x_matrix1 = np.reshape(x_matrix1, (nodes_number, 1))

                # normalization
                min_value, max_value = -435, -9
                x_matrix1 = np.subtract(x_matrix1, min_value)
                x_matrix1 = np.divide(x_matrix1, (max_value-min_value))
                print(x_matrix1.T)

                x_matrix1 = np.append(x_matrix1, 1).reshape(-1, 1)
                y_out = np.dot(weight_out_stm, x_matrix1)
                y_out_list.append(y_out[0, 0])
                x_final_matrix.append(x_matrix1.T.tolist()[0])
                envelope_matrix = []

        # debug
        plt.figure()
        plt.plot(envelope_matrix, c='red')
        # plt.show()

        # update weight
        y_train_matrix = np.array(train_signal).reshape(1, len(train_signal))
        x_final_matrix = np.asmatrix(x_final_matrix).T
        weight_out_stm = np.dot(y_train_matrix, np.linalg.pinv(x_final_matrix))

        # test for training
        y_out_test = np.dot(weight_out_stm, x_final_matrix)
        print(y_out_test.shape)
        print(y_out_test)
        plt.figure('PostProcessing')
        plt.plot(y_out_test)
        plt.show()

        # save weight matrix as .npy files
        np.save('weight_matrix_os/weight_out_classification.npy', weight_out_stm)

        # calculate the error
        error_learning = np.var(np.array(train_signal) - np.array(y_out_list))
        print('\rEpoch:{}/{}  error:{}'.format(j + 1, train_epoch, error_learning))

        print('Trained successfully !')
        print('----------------------------------------------------------------')

    return 0


def test_classification(a0, b0, c0, current_limited, test_number_wave, nodes_number):
    """
    a function used to test ability of classification task
    :param a0: initial value of magnetization at x axis
    :param b0: initial value of magnetization at y axis
    :param c0: initial value of magnetization at z axis
    :param current_limited: maximum and minimum value of applied current
    :param test_number_wave: the number of test wave signals
    :param nodes_number: size of reservoirs
    :return: no any return
    """
    # find the last weight matrix data
    if os.path.exists('weight_matrix_os/weight_out_classification.npy'):
        weight_out_stm = np.load('weight_matrix_os/weight_out_classification.npy')
        print('\r' + 'Loading weight_out_classification matrix successfully !', end='', flush=True)

    else:
        print('\rno valid weight data !', end='', flush=True)
        return 0

    print('\rClassification')
    print('----------------------------------------------------------------')
    print('start to test !', flush=True)

    # fabricate the input, target, and output signals
    train_epoch = 1
    for j in range(0, train_epoch):
        s_in, train_signal = waveform_generator(current_limited, test_number_wave)
        print('---------------------------------------------------------------')
        y_out_list = []
        m_x0, m_y0, m_z0 = a0, b0, c0
        envelope_matrix = []
        # used to get the figure

        # create pulse list
        for i1 in range(len(s_in)):
            mx_list, _, vol1, envelope_list, time_env_list, [m_x0, m_y0, m_z0], _ = do.evolution_mag(m_x0, m_y0, m_z0,
                                                                                                     magnitude=s_in[i1])
            # mx_list2, _, vol2, envelope_list2, time_env_list2, [m_x0, m_y0, m_z0], _ = do.evolution_mag(m_x0, m_y0,
            #                                                                                             m_z0,
            #                                                                                             magnitude=-i1)

            # envelope_list = envelope_list + envelope_list2
            envelope_matrix = envelope_matrix + envelope_list
            # sampling the nodes from resistances list
            if (i1+1) % 8 == 0 and i1 != 0:
                number_interval = int(len(envelope_matrix) / nodes_number)
                if number_interval < 1:
                    number_interval = 1
                x_matrix1 = np.array(envelope_matrix[1: len(envelope_matrix):number_interval])

                while len(x_matrix1) != nodes_number:
                    if len(x_matrix1) > nodes_number:
                        x_matrix1 = x_matrix1[:-1]
                    else:
                        x_matrix1 = np.append(x_matrix1, x_matrix1[-1])
                x_matrix1 = np.reshape(x_matrix1, (nodes_number, 1))

                # normalization
                min_value, max_value = -435, -9
                x_matrix1 = np.subtract(x_matrix1, min_value)
                x_matrix1 = np.divide(x_matrix1, (max_value - min_value))

                x_matrix1 = np.append(x_matrix1, 1).reshape(-1, 1)
                y_out = np.dot(weight_out_stm, x_matrix1)
                y_out_list.append(y_out[0, 0])
                envelope_matrix = []

        # calculate the error
        error_learning = np.var(np.array(train_signal) - np.array(y_out_list))
        print('\rEpoch:{}/{}  error:{}'.format(j + 1, train_epoch, error_learning))
        print('----------------------------------------------------------------')

        # FIGURES
        plt.figure('Train')
        plt.plot(envelope_matrix, c='blue')
        # plt.show()

        plt.figure('Test results')
        plt.plot(train_signal, c='blue', label='target')
        plt.plot(y_out_list, c='green', label='module')
        plt.ylabel('Index')
        plt.xlabel('Time')
        plt.legend()
        # plt.show()

        plt.figure('Comparison')
        plt.ylabel('Index')
        plt.xlabel('Time')
        plt.subplot(2, 1, 1)
        plt.plot(train_signal, c='blue', label='target')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(y_out_list, c='red', label='module')
        plt.legend()
        plt.show()

    return 0


if __name__ == '__main__':
    # for delay time task
    # train_stm(1, -0.01, 0.01, delay_time=1, nodes_number_stm=23)
    # test_stm(1, -0.01, 0.01, delay_time=1, test_points=100, nodes_number_stm=23)

    # for classification task
    train_classification(1, -0.01, 0.01, current_limited=0.8, number_wave=30, nodes_number=50)
    test_classification(1, -0.01, 0.01, current_limited=0.8, test_number_wave=30, nodes_number=50)
