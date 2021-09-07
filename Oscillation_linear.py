import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

# constant parameters
u0 = 12.56637e-7  # Vacuum permeability in H/m
h_bar = 1.054e-34  # Reduced Planck constant in Js
uB = 9.274e-24  # Bohr magneton in J/T
gamma = 2 * uB / (h_bar * 2 * np.pi)  # Gyromagnetic ratio in 1/Ts
alpha = 0.002
Hk = 0.5  # anisotropy field along x axis
Hd = 0.5  # demagnetization field along z axis
time = 2e-8  # simulation time
t_step = 1e-12  # time step
step = time / t_step
n = int(step)
R_ap = 400
R_p = 200

# constants of computing
Batch_size = 50
leaky_rate = 0.15
g_parameter = 1.5
seed = 1000
np.random.seed(seed)


def evolution_mag(m_x, m_y, m_z, direc_current=-0.836, magnitude=0.0, h_x=0.3, f_ac=8e11, time_used=int(n)):
    """
    a function used to build time evolution of MTJ oscillation
    :param m_x: initial state of magnetization vector
    :param m_y: initial state of magnetization vector
    :param m_z: initial state of magnetization vector
    :param direc_current: dc current used to maintain the oscillation
    :param magnitude: the magnitude of ac current
    :param h_x: external magnetic field
    :param f_ac: frequency of ac current
    :param time_used: time_step used to make evolution
    :return: mx_list, t_list, voltage_list, envelope_list1, time_env_list1, [mx, my, mz], my_list1
    """
    # initial
    t_list, resistance_list, voltage_list = [], [], []
    mx_list, my_list, mz_list = [], [], []
    # normalization
    norm = np.sqrt(pow(m_y, 2) + pow(m_x, 2) + pow(m_z, 2))
    mx = m_x / norm
    my = m_y / norm
    mz = m_z / norm

    # f_ac = 8e11

    for i1 in range(1, time_used):
        # analog_current = magnitude * np.sin(2 * np.pi * f_ac * i1 * t_step)  # sine function
        analog_current = magnitude
        sum_current = direc_current + analog_current

        H_DL = 2000 * sum_current  # field-like torque
        mx_1 = mx  # 0 = d_mx-gamma*Hd*my*mz-alpha*my*d_mz+alpha*mz*d_my-u0*gamma*H_DL*(my^2+mz^2);
        my_1 = my  # 0 = d_my+gamma*(Hx+Hk*mx+Hd*mx)*mz-alpha*mz*d_mx+alpha*mx*d_mz+u0*gamma*H_DL*mx*my;
        mz_1 = mz  # 0 = d_mz-gamma*(Hx+Hk*mx)*my-alpha*mx*d_my+alpha*my*d_mx+u0*gamma*H_DL*mx*mz;
        # k1
        A = gamma * Hd * my * mz
        B = u0 * gamma * H_DL * (my ** 2 + mz ** 2)
        C = -gamma * (h_x + Hk * mx + Hd * mx) * mz
        D = -u0 * gamma * H_DL * mx * my
        E = gamma * (h_x + Hk * mx) * my
        F = -u0 * gamma * H_DL * mx * mz
        a = 1 + (alpha ** 2) * (mz ** 2)
        b = alpha * my + (alpha ** 2) * mx * mz
        c = A + B - alpha * mz * (C + D)
        d = alpha * my - (alpha ** 2) * mx * mz
        e = 1 + (alpha ** 2) * (mx ** 2)
        f = E + F + alpha * mx * (C + D)
        d_mx1 = (b * f + c * e) / (a * e + b * d)
        d_mz1 = (a * f - c * d) / (a * e + b * d)
        d_my1 = C + D + alpha * mz * d_mx1 - alpha * mx * d_mz1
        mx = mx_1 + d_mx1 * t_step / 2
        my = my_1 + d_my1 * t_step / 2
        mz = mz_1 + d_mz1 * t_step / 2
        # k2
        A = gamma * Hd * my * mz
        B = u0 * gamma * H_DL * (my ** 2 + mz ** 2)
        C = -gamma * (h_x + Hk * mx + Hd * mx) * mz
        D = -u0 * gamma * H_DL * mx * my
        E = gamma * (h_x + Hk * mx) * my
        F = -u0 * gamma * H_DL * mx * mz
        a = 1 + (alpha ** 2) * (mz ** 2)
        b = alpha * my + (alpha ** 2) * mx * mz
        c = A + B - alpha * mz * (C + D)
        d = alpha * my - (alpha ** 2) * mx * mz
        e = 1 + (alpha ** 2) * (mx ** 2)
        f = E + F + alpha * mx * (C + D)
        d_mx2 = (b * f + c * e) / (a * e + b * d)
        d_mz2 = (a * f - c * d) / (a * e + b * d)
        d_my2 = C + D + alpha * mz * d_mx2 - alpha * mx * d_mz2
        mx = mx_1 + d_mx2 * t_step / 2
        my = my_1 + d_my2 * t_step / 2
        mz = mz_1 + d_mz2 * t_step / 2
        # k3
        A = gamma * Hd * my * mz
        B = u0 * gamma * H_DL * (my ** 2 + mz ** 2)
        C = -gamma * (h_x + Hk * mx + Hd * mx) * mz
        D = -u0 * gamma * H_DL * mx * my
        E = gamma * (h_x + Hk * mx) * my
        F = -u0 * gamma * H_DL * mx * mz
        a = 1 + (alpha ** 2) * (mz ** 2)
        b = alpha * my + (alpha ** 2) * mx * mz
        c = A + B - alpha * mz * (C + D)
        d = alpha * my - (alpha ** 2) * mx * mz
        e = 1 + (alpha ** 2) * (mx ** 2)
        f = E + F + alpha * mx * (C + D)
        d_mx3 = (b * f + c * e) / (a * e + b * d)
        d_mz3 = (a * f - c * d) / (a * e + b * d)
        d_my3 = C + D + alpha * mz * d_mx3 - alpha * mx * d_mz3
        mx = mx_1 + d_mx3 * t_step
        my = my_1 + d_my3 * t_step
        mz = mz_1 + d_mz3 * t_step
        # k4
        A = gamma * Hd * my * mz
        B = u0 * gamma * H_DL * (my ** 2 + mz ** 2)
        C = -gamma * (h_x + Hk * mx + Hd * mx) * mz
        D = -u0 * gamma * H_DL * mx * my
        E = gamma * (h_x + Hk * mx) * my
        F = -u0 * gamma * H_DL * mx * mz
        a = 1 + (alpha ** 2) * (mz ** 2)
        b = alpha * my + (alpha ** 2) * mx * mz
        c = A + B - alpha * mz * (C + D)
        d = alpha * my - (alpha ** 2) * mx * mz
        e = 1 + (alpha ** 2) * (mx ** 2)
        f = E + F + alpha * mx * (C + D)
        d_mx4 = (b * f + c * e) / (a * e + b * d)
        d_mz4 = (a * f - c * d) / (a * e + b * d)
        d_my4 = C + D + alpha * mz * d_mx4 - alpha * mx * d_mz4

        d_mx = d_mx1 + 2 * d_mx2 + 2 * d_mx3 + d_mx4
        d_my = d_my1 + 2 * d_my2 + 2 * d_my3 + d_my4
        d_mz = d_mz1 + 2 * d_mz2 + 2 * d_mz3 + d_mz4
        mx_1 = mx_1 + d_mx * t_step / 6
        my_1 = my_1 + d_my * t_step / 6
        mz_1 = mz_1 + d_mz * t_step / 6

        a = (mx_1 ** 2 + my_1 ** 2 + mz_1 ** 2) ** 0.5
        mx_1 = mx_1 / a
        my_1 = my_1 / a
        mz_1 = mz_1 / a

        mx = mx_1
        my = my_1
        mz = mz_1
        mx_list.append(mx)
        my_list.append(my)
        mz_list.append(mz)
        t_list.append(i1 * t_step * 1e9)

        theta = my
        resistance = 2 * R_p * R_ap / ((R_p + R_ap) + (R_ap - R_p) * theta)
        voltage_osc = resistance * sum_current
        resistance_list.append(resistance)
        voltage_list.append(voltage_osc)

    # calculate the resistance difference
    extreme_high_index, extreme_low_index = [], []
    for i in range(len(voltage_list)):
        if i != 0 and i != len(voltage_list) - 1:
            if voltage_list[i - 1] < voltage_list[i] and voltage_list[i] > voltage_list[i + 1]:
                extreme_high_index.append(voltage_list[i])
            elif voltage_list[i - 1] > voltage_list[i] and voltage_list[i] < voltage_list[i + 1]:
                extreme_low_index.append(voltage_list[i])

    resistance_dif = []
    for i in range(min(len(extreme_low_index), len(extreme_high_index))):
        temp = extreme_high_index[i] - extreme_low_index[i]
        resistance_dif.append(temp)

    return mx_list, t_list, voltage_list, [mx, my, mz], my_list, resistance_dif[-1]


def train_stm(a0, b0, c0, delay_time=1, nodes_number_stm=23, bias_stm=1, density=0.3, spectral_radius=0.2):
    """
    a function used to achieve training of classification by using oscillation MTJ
    :param a0: initial state of magnetization vector at x axis
    :param b0: initial state of magnetization vector at y axis
    :param c0: initial state of magnetization vector at z axis
    :param delay_time: delay time
    :param nodes_number_stm: number of nodes, reservoir size
    :param bias_stm: bias term
    :param density: the density of weight matrix between states
    :param spectral_radius: the largest elg_value of matrix
    :return: no any return
    """

    # initial weight matrix
    weight_state = scipy.sparse.rand(nodes_number_stm, nodes_number_stm, density=density, format='coo')
    elg_value, elg_vector = scipy.sparse.linalg.eigs(weight_state)
    weight_state = weight_state / max(np.abs(elg_value)) * spectral_radius
    weight_state = weight_state.todense()
    # input weight could be sparse, the function will be added in next version
    weight_feedback = np.random.rand(nodes_number_stm, 1) * 2 - 1

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
    s_in_stm = np.random.randint(0, 2, Batch_size)
    y_out_list, x_final_matrix = [], []
    last_mx, last_my, last_mz = [1]*nodes_number_stm, [0.01]*nodes_number_stm, [0.01]*nodes_number_stm
    current_states = [0]*nodes_number_stm
    input_list_reservoir = [0]*nodes_number_stm
    output = 0
    output_matrix = []
    state_final_matrix = []

    # create pulse list
    for i1 in range(0, Batch_size):
        if s_in_stm[i1] == 1:
            magnitude = 0.4
        else:
            magnitude = 0.8

        # neuron states update
        for i in range(nodes_number_stm):
            m_x0, m_y0, m_z0 = last_mx[i], last_my[i], last_mz[i]
            if i != 0:
                magnitude = input_list_reservoir[i]
            print(f'{m_x0}, {m_y0}, {m_z0}')
            mx_li1, t, v_1, [m_x1, m_y1, m_z1], my_li1, resist_dif = evolution_mag(m_x0, m_y0, m_z0,
                                                                                   magnitude=magnitude)
            # update states
            last_mx[i], last_my[i], last_mz[i] = m_x1, m_y1, m_z1
            current_states[i] = resist_dif
            # calculate the next term
            temp = 0
            for j in range(nodes_number_stm):
                if weight_state[j, i] != 0:
                    temp = weight_state[j, i]*current_states[j] + temp

            temp = g_parameter * temp
            output_current = (1-leaky_rate)*current_states[i] + leaky_rate*np.tanh(temp+weight_feedback[i]*output)

            for j in range(nodes_number_stm):
                if weight_state[i, j] != 0:
                    input_list_reservoir[j] = output_current

        # calculate the output
        current_states.append(1)  # bias
        current_states = np.array(current_states).reshape(1+nodes_number_stm, 1)
        output = np.dot(weight_out_stm, current_states)
        output_matrix.append(output)
        state_final_matrix.append(current_states.tolist()[0])

    # update matrix of readout part
    if delay_time != 0:
        y_train_signal = np.array(list(s_in_stm)[-int(delay_time):] + list(s_in_stm)[:-int(delay_time)])
    else:
        y_train_signal = s_in_stm
    state_final_matrix = np.asmatrix(state_final_matrix).reshape(-1, nodes_number_stm+1)
    weight_out_stm = np.dot(y_train_signal, state_final_matrix)
    print('Training successfully !')
    # save weight matrix as .npy files
    np.save('weight_matrix_os/weight_out_stm_{}.npy'.format(delay_time), weight_out_stm)

    return 0


if __name__ == '__main__':
    # mx_list, t_list, voltage_list, [mx, my, mz], my_list1, r_d = evolution_mag(1, 0.01, 0.01, magnitude=0.3)
    # plt.figure('evolution')
    # plt.plot(t_list, voltage_list)
    # plt.show()
    #
    # plt.figure('resistance')
    # plt.plot(r_d, c='red')
    # plt.show()

    # current_list = np.linspace(0, 1, 20)
    # resist_diff_list = []
    # for i in current_list:
    #     mx_list, t_list, vol_list, [mx, my, mz], _, r_d = evolution_mag(1, 0.01, 0.01, magnitude=i)
    #     # plt.figure()
    #     # plt.plot(t_list, vol_list)
    #     # plt.show()
    #     resist_diff_list.append(r_d)
    #
    # plt.figure('linear curve')
    # plt.plot(current_list, resist_diff_list)
    # plt.scatter(current_list, resist_diff_list, c='red')
    # plt.ylabel(r'$\Delta R$')
    # plt.xlabel('Current density')
    # plt.show()

    # for delay task
    train_stm(a0=1, b0=0.01, c0=0.01, delay_time=1, nodes_number_stm=30)
