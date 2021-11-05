import numpy as np
import matplotlib.pyplot as plt

# constant parameters
u0 = 12.56637e-7  # Vacuum permeability in H/m
h_bar = 1.054e-34  # Reduced Planck constant in Js
uB = 9.274e-24  # Bohr magneton in J/T
gamma = 2 * uB / (h_bar * 2 * np.pi)  # Gyromagnetic ratio in 1/Ts
alpha = 0.002
# H_x = 0.3  # applied filed along x axis
Hk = 0.5  # anisotropy field along x axis
Hd = 0.5  # demagnetization field along z axis
time = 2e-8  # simulation time
t_step = 1e-12  # time step
step = time / t_step
n = int(step)
R_ap = 400
R_p = 200


def evolution_mag(m_x, m_y, m_z, direc_current=-0.836, magnitude=0.0, h_x=0.3, f_ac=8e11, time_used=int(n/2)):
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
        analog_current = magnitude * np.sin(2 * np.pi * f_ac * i1 * t_step)  # sine function
        # analog_current = magnitude
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

    # finding the trajectory / envelope
    envelope_list, time_env_list = [], []
    for i in range(0, len(voltage_list)):
        if i != 0 and i != len(voltage_list) - 1:
            if voltage_list[i - 1] < voltage_list[i] and voltage_list[i] > voltage_list[i + 1]:
                envelope_list.append(voltage_list[i])
                time_env_list.append(t_list[i])

    # get the trajectory
    envelope_list1, time_env_list1 = [], []
    for i in range(len(envelope_list)):
        if i % 2 == 0:
            envelope_list1.append(envelope_list[i])
            time_env_list1.append(time_env_list[i])

    # # try one period:
    # envelope_list2, time_env_list2 = [], []
    # index_point_extreme = []
    # for ac_value in range(len(envelope_list1)):
    #     if ac_value != 0 and ac_value != len(envelope_list1) - 1:
    #         if envelope_list1[ac_value - 1] < envelope_list1[ac_value] and envelope_list1[ac_value] > envelope_list1[ac_value + 1]:
    #             # find the index of extreme point
    #             index_point_extreme.append(ac_value)

    # debug
    # print(len(index_point_extreme))

    # for ac_value in range(len(envelope_list1)):
    #     if index_point_extreme[1] >= ac_value >= index_point_extreme[0]:
    #         envelope_list2.append(envelope_list1[ac_value])
    #         time_env_list2.append(time_env_list1[ac_value])

    # debug
    # print(len(time_env_list2))

    # normalization
    # envelope_list1 = [ac_value/max(np.abs(envelope_list1)) for ac_value in envelope_list1]
    # voltage_list = [ac_value/max(np.abs(voltage_list)) for ac_value in voltage_list]
    return mx_list, t_list, voltage_list, envelope_list1, time_env_list1, [mx, my, mz], my_list


if __name__ == '__main__':
    mx_matrix, time_matrix, voltage_matrix, trace_list, trace_time, _, my_matrix = evolution_mag(1, -0.001, 0.001,
                                                                                                 direc_current=-0.836,
                                                                                                 magnitude=0.2,
                                                                                                 h_x=0.3, f_ac=8e11)

    plt.figure('Oscillation')
    plt.plot(time_matrix, mx_matrix, c='blue')
    plt.plot(time_matrix, my_matrix, c='red')
    plt.xlabel('time(ns)')
    plt.ylabel('mx')
    # plt.ylim(-1.2, 1.2)

    plt.figure()
    plt.plot(time_matrix, voltage_matrix)
    plt.scatter(trace_time, trace_list, c='red')
    plt.ylabel('Volt_ocs', fontsize=15)
    plt.xlabel('Time')
    plt.show()

    mx_matrix1, time_matrix1, voltage_matrix1, trace_list1, trace_time1, _, _ = evolution_mag(1, -0.001, 0.001,
                                                                                              direc_current=-0.836,
                                                                                              magnitude=0.1,
                                                                                              h_x=0.3, f_ac=8e11)
    plt.figure()
    plt.plot(time_matrix1, mx_matrix1)
    plt.xlabel('time(ns)')
    plt.ylabel('mx')

    plt.figure()
    plt.plot(time_matrix1, voltage_matrix1)
    plt.scatter(trace_time1, trace_list1, c='red')
    plt.ylabel('Voltage_ocs', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    # plt.show()

    plt.figure('comparison')
    plt.subplot(2, 1, 1)
    plt.title('+1 input', fontsize=12)
    plt.plot(trace_time, trace_list)
    plt.ylabel('Voltage', fontsize=12)

    plt.subplot(2, 1, 2)
    plt.title('-1 input', fontsize=12)
    plt.plot(trace_time1, trace_list1)
    plt.ylabel('Voltage', fontsize=12)
    plt.xlabel('time', fontsize=12)
    plt.show()

    # #############################################################################################################
    # dc current vs amplitude
    # ###############################################################################################################

    # amplitude_list = []
    # current_list = np.linspace(-11, 11, 50)
    # for ac_value in current_list:
    #     mx_matrix1, time_matrix1, voltage_matrix1, trace_list1, trace_time1, _, _ = evolution_mag(1, 0.01, 0.01,
    #                                                                                               direc_current=ac_value,
    #                                                                                               magnitude=0)
    #     if trace_list1:
    #         amplitude_list.append(np.max(trace_list1))
    #     else:
    #         amplitude_list.append(np.max(voltage_matrix1))
    #
    # plt.figure()
    # plt.plot(current_list, amplitude_list)
    # plt.scatter(current_list, amplitude_list)
    # plt.ylabel('Amplitude', fontsize=12)
    # plt.xlabel('dc current', fontsize=12)
    plt.show()
