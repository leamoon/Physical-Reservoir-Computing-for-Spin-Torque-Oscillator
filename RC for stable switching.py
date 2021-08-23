import numpy as np
import os
import device_module as dm
import matplotlib.pyplot as plt

# This code applies the virtual nodes method to build the reservoirs

# RC parameters
positive_voltage_stm = 0.7
negative_voltage_stm = -0.7
positive_voltage_pc = positive_voltage_stm
negative_voltage_pc = negative_voltage_stm
Batch_size = 9000  # batch size


# build a random network
def train_stm(a0, b0, c0, delay_time=1, nodes_number_stm=50, train_epoch=1, bias_stm=1):
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
            'weight_matrix/weight_out_stm_{}.npy'.format(delay_time)):
        weight_out_stm = np.load(
            'weight_matrix/weight_out_stm_{}.npy'.format(delay_time))
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
        s_in_stm = np.random.randint(0, 2, Batch_size)
        y_out_list = []
        x_final_matrix = []
        m_x0, m_y0, m_z0 = a0, b0, c0
        # create pulse list
        for i1 in range(0, Batch_size):
            if s_in_stm[i1] == 1:
                current_density = positive_voltage_stm
            else:
                current_density = negative_voltage_stm

            mx_list, m_x0, m_y0, m_z0, x_matrix1 = dm.module_evolution(50000, m_x=m_x0, m_y=m_y0, m_z=m_z0,
                                                                       current_density=current_density,
                                                                       sample_points=50)

            x_matrix1 = np.append(x_matrix1, bias_stm).reshape(-1, 1)
            y_out = np.dot(weight_out_stm, x_matrix1)
            y_out_list.append(y_out[0, 0])
            x_final_matrix.append(x_matrix1.T.tolist()[0])

            if i1 % 100 == 0:
                print('\r Process:{}/{}'.format(i1, Batch_size), end='', flush=True)

            # trick, maybe could be removed after introducing of thermal field
            if abs(m_x0) > 0.9:
                if abs(m_y0) < 0.01 or abs(m_z0) < 0.01:
                    m_y0, m_z0 = 0.01, 0.01
                    # normalization
                    mod_vector1 = np.sqrt(pow(m_x0, 2) + pow(m_y0, 2) + pow(m_z0, 2))
                    m_x0 = m_x0 / mod_vector1
                    m_y0 = m_y0 / mod_vector1
                    m_z0 = m_z0 / mod_vector1

        # update weight
        y_train_whole = list(s_in_stm)[-int(delay_time):] + list(s_in_stm)[0:-int(delay_time)]
        if delay_time == 0:
            y_train_whole = s_in_stm

        y_train_matrix = np.array(y_train_whole).reshape(1, len(y_train_whole))
        x_final_matrix = np.asmatrix(x_final_matrix).T
        weight_out_stm = np.dot(y_train_matrix, np.linalg.pinv(x_final_matrix))

        # save weight matrix as .npy files
        np.save('weight_matrix/weight_out_stm_{}.npy'.format(delay_time), weight_out_stm)

        # calculate the error
        error_learning = np.var(
            np.array(y_train_whole) - np.array(y_out_list))
        print('\rEpoch:{}/{}  error:{}'.format(j + 1, train_epoch, error_learning))

        print('Trained successfully !')
        print('----------------------------------------------------------------')
    return 0


def test_stm(a0, b0, c0, delay_time=1, test_points=20, bias_stm=1):
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
    if os.path.exists('weight_matrix/weight_out_stm_{}.npy'.format(delay_time)):
        weight_out_stm = np.load('weight_matrix/weight_out_stm_{}.npy'.format(delay_time))
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
            current_density = 0.7
        else:
            current_density = -0.7

        mx_list, m_x0, m_y0, m_z0, x_matrix1 = dm.module_evolution(50000, m_x=m_x0, m_y=m_y0, m_z=m_z0,
                                                                   current_density=current_density,
                                                                   sample_points=50)

        if abs(m_x0) > 0.9:
            if abs(m_y0) < 0.01 or abs(m_z0) < 0.01:
                m_y0, m_z0 = 0.01, 0.01
                # normalization
                mod_vector1 = np.sqrt(pow(m_x0, 2) + pow(m_y0, 2) + pow(m_z0, 2))
                m_x0 = m_x0 / mod_vector1
                m_y0 = m_y0 / mod_vector1
                m_z0 = m_z0 / mod_vector1

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

    plt.savefig('./FIGURES/train-delay={}'.format(delay_time))
    plt.close()
    # plt.show()

    # calculate the correlation
    cor_stm = pow(np.corrcoef(y_out_list[int(delay_time):], y_train_whole[int(delay_time):])[0, 1], 2)

    return cor_stm


def train_pc_task(a0, b0, c0, delay_time=1, nodes_number_pc=50, epoch_pc=1, bias_pc=1):
    """
    a function used to train weight matrix of Parity check task
    :param a0: initial value of magnetization at x axis
    :param b0: initial value of magnetization at y axis
    :param c0: initial value of magnetization at z axis
    :param delay_time: delay time
    :param nodes_number_pc: size of reservoirs
    :param epoch_pc: the number of training
    :param bias_pc: bias term, usually setting as 1
    :return: no any return, all of weights will be saved as .npy files
    """

    # finding existed weights
    if os.path.exists(
            'weight_matrix/weight_out_pc_{}.npy'.format(delay_time)):
        # weight_in_pc = np.load('weight_in_pc_{}.npy'.format(delay_time))
        # weight_mid_pc = np.load('weight_mid_pc_{}.npy'.format(delay_time))
        weight_out_pc = np.load(
            'weight_matrix/weight_out_pc_{}.npy'.format(delay_time))
        print('\rLoading weight_{} matrix of pc task successfully !'.format(delay_time), end='',
              flush=True)

    else:
        weight_out_pc = np.random.randint(-1, 2, (1, nodes_number_pc + 1))
        print('\rweight matrix of pc task has been created successfully !', end='', flush=True)

    print(f'\rPC  Time:{delay_time}')
    print('----------------------------------------------------------------')
    print('start to train !', flush=True)

    # initial values
    m_x0, m_y0, m_z0 = a0, b0, c0

    for j in range(0, epoch_pc):
        s_in_pc = np.random.randint(0, 2, Batch_size)

        y_out_list = []
        x_final_matrix = []
        # create pulse list
        for i1 in range(0, len(s_in_pc)):
            if s_in_pc[i1] == 1:
                current_density = positive_voltage_pc
            else:
                current_density = negative_voltage_pc

            mx_list, m_x0, m_y0, m_z0, x_matrix1 = dm.module_evolution(50000, m_x=m_x0, m_y=m_y0, m_z=m_z0,
                                                                       current_density=current_density,
                                                                       sample_points=50)

            x_matrix1 = np.append(x_matrix1, bias_pc).reshape(-1, 1)
            y_out = np.dot(weight_out_pc, x_matrix1)
            y_out_list.append(y_out[0, 0])
            x_final_matrix.append(x_matrix1.T.tolist()[0])
            if i1 % 100 == 0:
                print('\rProcess:{}/{}'.format(i1, Batch_size), end='', flush=True)

            # trick
            if abs(m_x0) > 0.9:
                if abs(m_y0) < 0.01 or abs(m_z0) < 0.01:
                    m_y0, m_z0 = 0.01, 0.01
                    # normalization
                    mod_vector1 = np.sqrt(pow(m_x0, 2) + pow(m_y0, 2) + pow(m_z0, 2))
                    m_x0 = m_x0 / mod_vector1
                    m_y0 = m_y0 / mod_vector1
                    m_z0 = m_z0 / mod_vector1

        # build the train_pc pulse
        y_train_pc = s_in_pc
        if delay_time != 0:
            for i2 in range(1, delay_time + 1):
                temp_var1 = np.array(list(s_in_pc)[-int(i2):] + list(s_in_pc)[0:-i2])
                y_train_pc = temp_var1 + y_train_pc
                y_train_pc[y_train_pc == 2] = 0

        y_train_matrix = np.array(y_train_pc).reshape(1, len(y_train_pc))
        x_final_matrix = np.asmatrix(x_final_matrix).T
        weight_out_pc = np.dot(y_train_matrix, np.linalg.pinv(x_final_matrix))
        np.save('weight_matrix/weight_out_pc_{}.npy'.format(delay_time), weight_out_pc)
        # calculate the error
        error_learning = np.var(np.array(y_train_pc) - np.array(y_out_list))
        print('\rEpoch:{}/{}  error:{}'.format(j + 1, epoch_pc, error_learning), flush=True)
        print('Trained successfully !')
        print('----------------------------------------------------------------')

    return 0


def test_pc_task(a0, b0, c0, delay_time=1, test_points=100, bias_pc=1):
    """
    a function used to test the ability of Parity Check Task of neuron network
    :param a0: initial value of magnetization at x axis
    :param b0: initial value of magnetization at y axis
    :param c0: initial value of magnetization at z axis
    :param delay_time: delay time
    :param test_points: the length of test_input signals
    :param bias_pc: bias term
    :return: cor_pc, the correlation of target and actual output
    """

    # loading weights
    if os.path.exists('weight_matrix/weight_out_pc_{}.npy'.format(delay_time)):
        # weight_in_pc = np.load('weight_in_pc.npy')
        # weight_mid_pc = np.load('weight_mid_pc.npy')
        weight_out_pc = np.load('weight_matrix/weight_out_pc_{}.npy'.format(delay_time))
        print('Loading weight_pc_{} matrix of pc task successfully !'.format(delay_time))

    else:
        print('the pc weight_pc_{} matrix does not exist !'.format(delay_time))
        return 0

    y_out_list = []
    # product test data
    s_in_pc = np.random.randint(0, 2, test_points)
    m_x0, m_y0, m_z0 = a0, b0, c0

    # create pulse list
    for i1 in range(0, len(s_in_pc)):
        if s_in_pc[i1] == 1:
            current_density = positive_voltage_pc
        else:
            current_density = negative_voltage_pc

        mx_list, m_x0, m_y0, m_z0, x_matrix1 = dm.module_evolution(50000, m_x=m_x0, m_y=m_y0, m_z=m_z0,
                                                                   current_density=current_density,
                                                                   sample_points=50)

        x_matrix1 = np.append(x_matrix1, bias_pc).reshape(-1, 1)
        y_out = np.dot(weight_out_pc, x_matrix1)
        y_out_list.append(y_out[0, 0])

        # trick
        if abs(m_x0) > 0.9:
            if abs(m_y0) < 0.01 or abs(m_z0) < 0.01:
                m_y0, m_z0 = 0.01, 0.01
                # normalization
                mod_vector1 = np.sqrt(pow(m_x0, 2) + pow(m_y0, 2) + pow(m_z0, 2))
                m_x0 = m_x0 / mod_vector1
                m_y0 = m_y0 / mod_vector1
                m_z0 = m_z0 / mod_vector1

    y_test_pc = s_in_pc
    if delay_time != 0:
        for i2 in range(1, delay_time + 1):
            temp_var1 = np.array(list(s_in_pc)[-int(i2):] + list(s_in_pc)[0:-i2])
            y_test_pc = temp_var1 + y_test_pc
            y_test_pc[y_test_pc == 2] = 0

    error_learning = np.var(np.array(y_test_pc) - np.array(y_out_list))
    print('error:{}'.format(error_learning))

    t_step = list(range(0, len(s_in_pc)))
    t_step = [i1 * 20e-9 for i1 in t_step]
    t_step_delay = list(range(0, len(s_in_pc)))
    t_step_delay = [i1 * 20e-9 for i1 in t_step_delay]

    # FIGURE
    plt.figure('PC-task')
    plt.subplot(3, 1, 1)
    plt.plot(t_step[10:], s_in_pc[10:])
    plt.xlabel('T_step')
    plt.ylabel('S_in')
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t_step_delay[10:], y_test_pc[10:])
    plt.xlabel('T_step')
    plt.ylabel('y_train_PC')
    plt.grid()

    plt.subplot(3, 1, 3)
    print('y_out:{}'.format(y_out_list))
    plt.plot(t_step[10:], y_out_list[10:])
    plt.xlabel('T_step')
    plt.ylabel('y_out_PC')

    plt.savefig('./FIGURES/PC-train-delay={}'.format(delay_time))
    plt.close()
    # plt.show()

    # calculate the correlation
    cor_pc = pow(np.corrcoef(y_out_list[int(delay_time):], y_test_pc[int(delay_time):])[0, 1], 2)
    return cor_pc


if __name__ == '__main__':
    m_x, m_y, m_z = 1, 0.01, 0.01
    # for delay time task without demagnetization
    # train_stm(m_x, m_y, m_z, delay_time=2, nodes_number_stm=50)
    # test_stm(m_x, m_y, m_z, delay_time=2, test_points=100)

    # for PC task without demagnetization
    # train_pc_task(m_x, m_y, m_z, delay_time=1, nodes_number_pc=50)
    # test_pc_task(m_x, m_y, m_z, delay_time=1, test_points=100)

    ################################################################################################################
    # a test of comparison
    # test code
    # capacity_stm, capacity_pc = [], []
    # for i in range(0, 7):
    #     cor1 = test_stm(m_x, m_y, m_z, delay_time=i, test_points=100)
    #     capacity_stm.append(cor1)
    #     print('---------------------------------------')
    #     print(f'delay time = {i}  STM!')
    #     print('---------------------------------------')
    #
    #     cor2 = test_pc_task(m_x, m_y, m_z, delay_time=i, test_points=100)
    #     capacity_pc.append(cor2)
    #     print('---------------------------------------')
    #     print(f'delay time = {i}  PC !')
    #     print('---------------------------------------')

    # capacity
    # print('capacity_stm:{}'.format(capacity_stm))
    # print('capacity_pc:{}'.format(capacity_pc))

    # results
    capacity_stm = [0.9999999994805995, 0.9999996330867625, 0.9999412769289289, 0.1669441968817252,
                    0.04968739491271108, 0.0007354314653622022, 0.022378047756496006]
    capacity_pc = [0.9999999997142932, 0.9999999969787305, 0.9999251613141497, 0.1888653184046998, 0.0597141768665486,
                   0.0017979232561799695, 0.010448308224999862]

    print('capacity_STM:{}'.format(sum([i * i for i in capacity_stm])))
    print('capacity_PC:{}'.format(sum([i * i for i in capacity_pc])))

    delay_time_list = np.linspace(0, 6, 7)
    plt.figure('Capacity')
    plt.subplot(2, 1, 1)
    plt.plot(delay_time_list, capacity_stm, label='STM')
    plt.scatter(delay_time_list, capacity_stm)
    plt.fill_between(delay_time_list, capacity_stm, alpha=0.4)
    plt.text(1, 0.5, 'Capacity={:.2f}'.format(sum([i * i for i in capacity_stm])), c='blue')
    plt.title('delay time task')
    plt.ylabel(r'quality')

    plt.subplot(2, 1, 2)
    plt.plot(delay_time_list, capacity_pc, label='PC')
    plt.scatter(delay_time_list, capacity_pc)
    plt.fill_between(delay_time_list, capacity_pc, alpha=0.5)
    plt.text(1, 0.5, 'Capacity={:.2f}'.format(sum([i * i for i in capacity_pc])), c='blue')
    plt.xlabel(r'delay time')
    plt.ylabel(r'quality')
    plt.title('Parity check task')
    plt.show()
