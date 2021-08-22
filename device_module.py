import numpy as np
import matplotlib.pyplot as plt
from pyESN import ESN

# constant parameters
u0 = 12.56637e-7  # Vacuum permeability in H/m
h_bar = 1.054e-34  # Reduced Planck constant in Js
K_B = 1.38e-23  # Boltzmann's constant
uB = 9.274e-24  # Bohr magneton in J/T
gamma = 2 * uB / (h_bar * 2 * np.pi)  # Gyromagnetic ratio in 1/Ts
T = 300  # temperature

# parameters of devices
alpha = 0.002  # damping factor
H_x = 0.1  # applied filed along x axis
H_k = 0.2  # anisotropy field along x axis
H_d = 0  # demagnetization field along z axis          0.4
j = 0.7  # DC current density in MA/cm^2
time_end = 0.2e-6  # simulation time 1e-6
t_step = 1e-11  # time step
step = int(time_end / t_step)
M_s = 1  # saturation magnetization
Volume = 1  # the volume of FL
H_th = np.sqrt(2*K_B*T*alpha/(u0*t_step*gamma*M_s*Volume))
H_th = 0
print(H_th)


def module_evolution(time_step, m_x, m_y, m_z, current_density, sample_points=50):
    """
    :param time_step: the step of time evolution, i1.e. the number of sampling points
    :param m_x: initial state at x axis
    :param m_y: initial state at y axis
    :param m_z: initial state at z axis
    :param current_density: the input current density
    :param sample_points: a number, produce a points list for usage of RC learning
    :return: the matrix of all of m_x, the last state of m vector, virtual matrix
    """
    # initial states for all of matrix
    time_step = int(time_step)
    matrix_mx = []
    matrix_my = []
    matrix_mz = []
    mx_initial = m_x

    for i1 in range(1, time_step):
        # t = i1 * t_step
        h_dl = 2000 * current_density  # field-like torque
        mx_1, my_1, mz_1 = m_x, m_y, m_z

        # normalization
        mod_vector = np.sqrt(pow(mx_1, 2) + pow(my_1, 2) + pow(mz_1, 2))
        mx_1 = mx_1 / mod_vector
        my_1 = my_1 / mod_vector
        mz_1 = mz_1 / mod_vector

        matrix_mx.append(m_x)
        matrix_my.append(m_y)
        matrix_mz.append(m_z)

        # k1 term
        b = alpha * m_y + (alpha ** 2) * m_x * m_z
        c = gamma * H_d * m_y * m_z + u0 * gamma * h_dl * (m_y ** 2 + m_z ** 2) - \
            alpha * m_z * (-gamma * (H_x + H_k * m_x + H_d * m_x + H_th) * m_z - u0 * gamma * h_dl * m_x * m_y)
        d = alpha * m_y - (alpha ** 2) * m_x * m_z
        e = 1 + (alpha ** 2) * (m_x ** 2)
        f = gamma * (H_x + H_k * m_x+H_th) * m_y - u0 * gamma * h_dl * m_x * m_z + alpha * m_x * (
                -gamma * (H_x + H_k * m_x + H_d * m_x+ H_th) * m_z - u0 * gamma * h_dl * m_x * m_y)
        d_mx1 = (b * f + c * e) / ((1 + (alpha ** 2) * (m_z ** 2)) * e + b * d)
        d_mz1 = ((1 + (alpha ** 2) * (m_z ** 2)) * f - c * d) / ((1 + (alpha ** 2) * (m_z ** 2)) * e + b * d)
        d_my1 = -gamma * (H_x + H_k * m_x + H_d * m_x+H_th) * m_z - u0 * gamma * h_dl * m_x * m_y + alpha * m_z * d_mx1 \
                - alpha * m_x * d_mz1
        m_x = mx_1 + d_mx1 * t_step / 2
        m_y = my_1 + d_my1 * t_step / 2
        m_z = mz_1 + d_mz1 * t_step / 2
        # k2
        A = gamma * H_d * m_y * m_z
        B = u0 * gamma * h_dl * (m_y ** 2 + m_z ** 2)
        C = -gamma * (H_x + H_k * m_x + H_d * m_x + H_th) * m_z
        D = -u0 * gamma * h_dl * m_x * m_y
        E = gamma * (H_x + H_k * m_x + H_th) * m_y
        F = -u0 * gamma * h_dl * m_x * m_z
        a = 1 + (alpha ** 2) * (m_z ** 2)
        b = alpha * m_y + (alpha ** 2) * m_x * m_z
        c = A + B - alpha * m_z * (C + D)
        d = alpha * m_y - (alpha ** 2) * m_x * m_z
        e = 1 + (alpha ** 2) * (m_x ** 2)
        f = E + F + alpha * m_x * (C + D)
        d_mx2 = (b * f + c * e) / (a * e + b * d)
        d_mz2 = (a * f - c * d) / (a * e + b * d)
        d_my2 = C + D + alpha * m_z * d_mx2 - alpha * m_x * d_mz2
        m_x = mx_1 + d_mx2 * t_step / 2
        m_y = my_1 + d_my2 * t_step / 2
        m_z = mz_1 + d_mz2 * t_step / 2
        # k3
        A = gamma * H_d * m_y * m_z
        B = u0 * gamma * h_dl * (m_y ** 2 + m_z ** 2)
        C = -gamma * (H_x + H_k * m_x + H_d * m_x + H_th) * m_z
        D = -u0 * gamma * h_dl * m_x * m_y
        E = gamma * (H_x + H_k * m_x + H_th) * m_y
        F = -u0 * gamma * h_dl * m_x * m_z
        a = 1 + (alpha ** 2) * (m_z ** 2)
        b = alpha * m_y + (alpha ** 2) * m_x * m_z
        c = A + B - alpha * m_z * (C + D)
        d = alpha * m_y - (alpha ** 2) * m_x * m_z
        e = 1 + (alpha ** 2) * (m_x ** 2)
        f = E + F + alpha * m_x * (C + D)
        d_mx3 = (b * f + c * e) / (a * e + b * d)
        d_mz3 = (a * f - c * d) / (a * e + b * d)
        d_my3 = C + D + alpha * m_z * d_mx3 - alpha * m_x * d_mz3
        m_x = mx_1 + d_mx3 * t_step
        m_y = my_1 + d_my3 * t_step
        m_z = mz_1 + d_mz3 * t_step
        # k4
        A = gamma * H_d * m_y * m_z
        B = u0 * gamma * h_dl * (m_y ** 2 + m_z ** 2)
        C = -gamma * (H_x + H_k * m_x + H_d * m_x + H_th) * m_z
        D = -u0 * gamma * h_dl * m_x * m_y
        E = gamma * (H_x + H_k * m_x + H_th) * m_y
        F = -u0 * gamma * h_dl * m_x * m_z
        a = 1 + (alpha ** 2) * (m_z ** 2)
        b = alpha * m_y + (alpha ** 2) * m_x * m_z
        c = A + B - alpha * m_z * (C + D)
        d = alpha * m_y - (alpha ** 2) * m_x * m_z
        e = 1 + (alpha ** 2) * (m_x ** 2)
        f = E + F + alpha * m_x * (C + D)
        d_mx4 = (b * f + c * e) / (a * e + b * d)
        d_mz4 = (a * f - c * d) / (a * e + b * d)
        d_my4 = C + D + alpha * m_z * d_mx4 - alpha * m_x * d_mz4

        # the classical fourth Runge Kutta method
        d_mx = d_mx1 + 2 * d_mx2 + 2 * d_mx3 + d_mx4
        d_my = d_my1 + 2 * d_my2 + 2 * d_my3 + d_my4
        d_mz = d_mz1 + 2 * d_mz2 + 2 * d_mz3 + d_mz4
        mx_1 = mx_1 + d_mx * t_step / 6
        my_1 = my_1 + d_my * t_step / 6
        mz_1 = mz_1 + d_mz * t_step / 6

        # trick
        if mx_1 > 0 > mx_initial:
            if abs(mx_1) > 0.98:
                if abs(mz_1) < 0.01 or abs(my_1) < 0.01:
                    break

        if mx_1 < 0 < mx_initial:
            if abs(mx_1) > 0.98:
                if abs(mz_1) < 0.01 or abs(my_1) < 0.01:
                    break
    # virtual nodes
    # sampling the nodes from resistances list
    number_interval = int(len(matrix_mx) / sample_points)
    virtual_matrix = np.array(matrix_mx[1: len(matrix_mx):number_interval])

    while len(virtual_matrix) != sample_points:
        if len(virtual_matrix) > sample_points:
            virtual_matrix = virtual_matrix[:-1]
        else:
            virtual_matrix = list(virtual_matrix) + list(virtual_matrix)[-1]
            virtual_matrix = np.array(virtual_matrix)
    virtual_matrix = np.reshape(virtual_matrix, (sample_points, 1))

    return matrix_mx, mx_1, my_1, mz_1, virtual_matrix


if __name__ == '__main__':
    input_signals = list(np.random.randint(0, 2, 10))
    input_voltage = []
    for i in input_signals:
        if i == 1:
            input_voltage.append(-0.7)
        else:
            input_voltage.append(0.7)

    m_x0, m_y0, m_z0 = 1, 0.01, 0.01  # initial values
    mx_matrix = []
    virtual_matrix, t_step_matrix = [], []
    input_voltage = [-0.7]
    print(input_voltage)
    for i in input_voltage:
        mx_list, m_x0, m_y0, m_z0, virtual_nodes = module_evolution(50000, m_x=m_x0, m_y=m_y0, m_z=m_z0,
                                                                    current_density=i)
        mx_matrix = mx_matrix + list(mx_list[:-1])
        virtual_matrix = virtual_matrix + list(virtual_nodes)
        print('last_state: mx, my, mz = {}, {}, {}'.format(m_x0, m_y0, m_z0))

        # trick
        if abs(m_x0) > 0.9:
            if abs(m_y0) < 0.01 or abs(m_z0) < 0.01:
                m_y0, m_z0 = 0.01, 0.01
                # normalization
                mod_vector1 = np.sqrt(pow(m_x0, 2) + pow(m_y0, 2) + pow(m_z0, 2))
                m_x0 = m_x0 / mod_vector1
                m_y0 = m_y0 / mod_vector1
                m_z0 = m_z0 / mod_vector1

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
    #     if abs(m_x0) > 0.9:
    #         if abs(m_y0) < 0.01 or abs(m_z0) < 0.01:
    #             m_y0, m_z0 = 0.01, 0.01
    #             # normalization
    #             mod_vector1 = np.sqrt(pow(m_x0, 2) + pow(m_y0, 2) + pow(m_z0, 2))
    #             m_x0 = m_x0 / mod_vector1
    #             m_y0 = m_y0 / mod_vector1
    #             m_z0 = m_z0 / mod_vector1
    #     plt.figure()
    #     plt.plot(mx_matrix)
    # plt.show()
