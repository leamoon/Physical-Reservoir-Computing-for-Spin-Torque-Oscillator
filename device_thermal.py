import numpy as np
import matplotlib.pyplot as plt

# constant parameters
u0 = 12.56637e-7  # Vacuum permeability in H/m
h_bar = 1.054e-34  # Reduced Planck constant in Js
K_B = 1.38e-23  # Boltzmann's constant
uB = 9.274e-24  # Bohr magneton in J/T
gamma = 2 * uB / (h_bar * 2 * np.pi)  # Gyromagnetic ratio in 1/Ts

# parameters of devices
alpha = 0.002  # damping factor
H_x = 0.3  # applied filed along x axis
H_k = 0.3  # anisotropy field along x axis
H_d = 1.2  # demagnetization field along z axis          0.4
# j = 0.45  # DC current density in MA/cm^2
time_end = 0.2e-6  # simulation time 1e-6
t_step = 1e-11  # time step
step = int(time_end / t_step)


def module_evolution(time_step, mx, my, mz, current_density, sample_points=50):
    """
    :param time_step: the step of time evolution, i1.e. the number of sampling points
    :param mx: initial state at x axis
    :param my: initial state at y axis
    :param mz: initial state at z axis
    :param current_density: the input current density
    :param sample_points: a number, produce a points list for usage of RC learning
    :return: the matrix of all of m_x, the last state of m vector, virtual matrix
    """
    # initial states for all of matrix
    time_step = int(time_step)
    matrix_mx = []
    matrix_my = []
    matrix_mz = []

    for i1 in range(1, time_step):
        # t = i1 * t_step
        h_dl = 2000 * current_density  # field-like torque
        mx_1, my_1, mz_1 = mx, my, mz

        # normalization
        mod_vector = np.sqrt(pow(mx_1, 2) + pow(my_1, 2) + pow(mz_1, 2))
        mx_1 = mx_1 / mod_vector
        my_1 = my_1 / mod_vector
        mz_1 = mz_1 / mod_vector

        # results list
        matrix_mx.append(mx_1)
        matrix_my.append(my_1)
        matrix_mz.append(mz_1)

        N = 0  # thermal noise, 0 off, 1 on
        sigma_x = 0
        sigma_y = 0
        sigma_z = 0
        H_TH = 0.0005  # thermal field
        if N == 1:
            sigma_x = 2 * np.random.random() - 1
            sigma_y = 2 * np.random.random() - 1
            sigma_z = 2 * np.random.random() - 1
        Heff_x = H_x + H_k * mx + N * sigma_x * H_TH
        Heff_y = N * sigma_y * H_TH
        Heff_z = -H_d * mz + N * sigma_z * H_TH
        # k1
        A = gamma * Heff_y * mz - gamma * Heff_z * my
        B = u0 * gamma * h_dl * (my ** 2 + mz ** 2)
        C = gamma * Heff_z * mx - gamma * Heff_x * mz
        D = -u0 * gamma * h_dl * mx * my
        E = gamma * Heff_x * my - gamma * Heff_y * mx
        F = -u0 * gamma * h_dl * mx * mz
        a = 1 + alpha ** 2 * mz ** 2
        b = alpha * my + alpha ** 2 * mx * mz
        c = A + B - alpha * mz * (C + D)
        d = alpha * my - alpha ** 2 * mx * mz
        e = 1 + alpha ** 2 * mx ** 2
        f = E + F + alpha * mx * (C + D)
        d_mx1 = (b * f + c * e) / (a * e + b * d)
        d_mz1 = (a * f - c * d) / (a * e + b * d)
        d_my1 = C + D + alpha * mz * d_mx1 - alpha * mx * d_mz1
        mx = mx_1 + d_mx1 * t_step / 2
        my = my_1 + d_my1 * t_step / 2
        mz = mz_1 + d_mz1 * t_step / 2
        # k2
        Heff_x = H_x + H_k * mx + N * sigma_x * H_TH
        Heff_y = N * sigma_y * H_TH
        Heff_z = -H_d * mz + N * sigma_z * H_TH
        A = gamma * Heff_y * mz - gamma * Heff_z * my
        B = u0 * gamma * h_dl * (my ** 2 + mz ** 2)
        C = gamma * Heff_z * mx - gamma * Heff_x * mz
        D = -u0 * gamma * h_dl * mx * my
        E = gamma * Heff_x * my - gamma * Heff_y * mx
        F = -u0 * gamma * h_dl * mx * mz
        a = 1 + alpha ** 2 * mz ** 2
        b = alpha * my + alpha ** 2 * mx * mz
        c = A + B - alpha * mz * (C + D)
        d = alpha * my - alpha ** 2 * mx * mz
        e = 1 + alpha ** 2 * mx ** 2
        f = E + F + alpha * mx * (C + D)
        d_mx2 = (b * f + c * e) / (a * e + b * d)
        d_mz2 = (a * f - c * d) / (a * e + b * d)
        d_my2 = C + D + alpha * mz * d_mx2 - alpha * mx * d_mz2
        mx = mx_1 + d_mx2 * t_step / 2
        my = my_1 + d_my2 * t_step / 2
        mz = mz_1 + d_mz2 * t_step / 2
        # k3
        Heff_x = H_x + H_k * mx + N * sigma_x * H_TH
        Heff_y = N * sigma_y * H_TH
        Heff_z = -H_d * mz + N * sigma_z * H_TH
        A = gamma * Heff_y * mz - gamma * Heff_z * my
        B = u0 * gamma * h_dl * (my ** 2 + mz ** 2)
        C = gamma * Heff_z * mx - gamma * Heff_x * mz
        D = -u0 * gamma * h_dl * mx * my
        E = gamma * Heff_x * my - gamma * Heff_y * mx
        F = -u0 * gamma * h_dl * mx * mz
        a = 1 + alpha ** 2 * mz ** 2
        b = alpha * my + alpha ** 2 * mx * mz
        c = A + B - alpha * mz * (C + D)
        d = alpha * my - alpha ** 2 * mx * mz
        e = 1 + alpha ** 2 * mx ** 2
        f = E + F + alpha * mx * (C + D)
        d_mx3 = (b * f + c * e) / (a * e + b * d)
        d_mz3 = (a * f - c * d) / (a * e + b * d)
        d_my3 = C + D + alpha * mz * d_mx3 - alpha * mx * d_mz3
        mx = mx_1 + d_mx3 * t_step
        my = my_1 + d_my3 * t_step
        mz = mz_1 + d_mz3 * t_step
        # k4
        Heff_x = H_x + H_k * mx + N * sigma_x * H_TH
        Heff_y = N * sigma_y * H_TH
        Heff_z = -H_d * mz + N * sigma_z * H_TH
        A = gamma * Heff_y * mz - gamma * Heff_z * my
        B = u0 * gamma * h_dl * (my ** 2 + mz ** 2)
        C = gamma * Heff_z * mx - gamma * Heff_x * mz
        D = -u0 * gamma * h_dl * mx * my
        E = gamma * Heff_x * my - gamma * Heff_y * mx
        F = -u0 * gamma * h_dl * mx * mz
        a = 1 + alpha ** 2 * mz ** 2
        b = alpha * my + alpha ** 2 * mx * mz
        c = A + B - alpha * mz * (C + D)
        d = alpha * my - alpha ** 2 * mx * mz
        e = 1 + alpha ** 2 * mx ** 2
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

    # virtual nodes
    # sampling the nodes from resistances list
    number_interval = int(len(matrix_mx) / sample_points)
    virtual_matrix1 = np.array(matrix_mx[1: len(matrix_mx):number_interval])

    while len(virtual_matrix1) != sample_points:
        if len(virtual_matrix1) > sample_points:
            virtual_matrix1 = virtual_matrix1[:-1]
        else:
            virtual_matrix1 = list(virtual_matrix1) + list(virtual_matrix1)[-1]
            virtual_matrix1 = np.array(virtual_matrix1)
    virtual_matrix1 = np.reshape(virtual_matrix1, (sample_points, 1))

    return matrix_mx, mx_1, my_1, mz_1, virtual_matrix1


if __name__ == '__main__':
    input_signals = list(np.random.randint(0, 2, 30))
    input_voltage = []
    for i in input_signals:
        if i == 1:
            input_voltage.append(-1)
        else:
            input_voltage.append(-1)

    m_x0, m_y0, m_z0 = 0, 0.1, 0.1  # initial values
    mx_matrix = []
    virtual_matrix, t_step_matrix = [], []
    input_voltage = [-1]
    print(input_voltage)
    for i in input_voltage:
        mx_list, m_x0, m_y0, m_z0, virtual_nodes = module_evolution(500000, mx=m_x0, my=m_y0, mz=m_z0,
                                                                    current_density=i)
        mx_matrix = mx_matrix + list(mx_list[:-1])
        virtual_matrix = virtual_matrix + list(virtual_nodes)
        print('last_state: mx, my, mz = {}, {}, {}'.format(m_x0, m_y0, m_z0))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.ylabel(r'input_signals')
    plt.xlabel('time')
    plt.plot(input_voltage)

    plt.subplot(2, 1, 2)
    time_interval = np.linspace(0, time_end * len(input_signals), len(mx_matrix))
    plt.plot(time_interval, mx_matrix)
    t1 = time_interval[0:len(time_interval):1000]
    p1 = mx_matrix[0:len(mx_matrix):1000]
    # plt.scatter(t1, p1, c='red', s=len(t1))
    # print(len(time_interval))
    # print(len(t1))
    plt.ylim(-1.1, 1.1)
    plt.ylabel(r'm_x')
    plt.xlabel('time')
    plt.show()

    # test for M-current density curve
    # input_density = 0.1*np.random.randint(-8, 8, 10)
    # input_density = [-0.2, -0.3, -0.4, -0.5]
    # print(input_density)
    # m_x0, m_y0, m_z0 = 1, 0.1, 0.1
    # # mx_matrix = []
    # for i in input_density:
    #     mx_matrix = []
    #     m_x0, m_y0, m_z0 = 1, 0.1, 0.1
    #     mx_list, m_x0, m_y0, m_z0, _ = module_evolution(30000, m_x=m_x0, m_y=m_y0, m_z=m_z0, current_density=i)
    #     mx_matrix = mx_matrix + list(mx_list[:-1])
    #     print('last_state: mx, my, mz = {}, {}, {}'.format(m_x0, m_y0, m_z0))
    #
    #     plt.figure()
    #     plt.plot(mx_matrix)
    # plt.show()
