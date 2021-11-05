import os.path
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from rich.progress import track
import pandas as pd
from matplotlib.pyplot import MultipleLocator

"""
this code is designed to calculate the lyapunov exponent function of Lorenz system.

We can package this function to judge edge of chaos in spintronic system.

The Lorenz System:

partial_x = sigma * (y - x)
partial_y = r*x - y - x*z
partial_z = x*y - batter*z

"""


# def step_evolution(x0, y0, z0, r, t_step=0.01):
#     # fourth Runge-Kutta
#     sigma, beta = 10, 8/3
#     h = t_step
#
#     # k1
#     k1_x = sigma*(y0-x0)
#     k1_y = r*x0 - y0 - x0*z0
#     k1_z = x0*y0 - beta*z0
#
#     # k2
#     k2_x = sigma*(y0-x0)
#     k2_y = r*(x0 + h/2*k1_x) - (y0+h/2*k1_y) - (x0+h/2*k1_x)*(z0+h/2*k1_z)
#     k2_z = (x0 + h/2*k1_x)*(y0+h/2*k1_y) - beta*(z0+h/2*k1_z)
#
#     # k3
#     k3_x = sigma * (y0 - x0)
#     k3_y = r * (x0 + h / 2 * k2_x) - (y0 + h / 2 * k2_y) - (x0 + h / 2 * k2_x) * (z0 + h / 2 * k2_z)
#     k3_z = (x0 + h / 2 * k2_x) * (y0 + h / 2 * k2_y) - beta * (z0 + h / 2 * k2_z)
#
#     # k4
#     k4_x = sigma * (y0 - x0)
#     k4_y = r * (x0 + h * k3_x) - (y0 + h * k3_y) - (x0 + h * k3_x) * (z0 + h * k3_z)
#     k4_z = (x0 + h * k3_x) * (y0 + h * k3_y) - beta * (z0 + h * k3_z)
#
#     x_new = x0 + h/6*(k1_x + 2*k2_x + 2*k3_x + k4_x)
#     y_new = y0 + h/6*(k1_y + 2*k2_y + 2*k3_y + k4_y)
#     z_new = z0 + h/6*(k1_z + 2*k2_z + 2*k3_z + k4_z)
#     return x_new, y_new, z_new


# def time_evolution(x0=0.1, y0=0, z0=0, r=40, time_consume=0.1, t_step=0.0001):
#     number_interval = int(time_consume/t_step)
#     x_list, y_list, z_list = [], [], []
#     x, y, z = x0, y0, z0
#     for ac_value in range(number_interval):
#         x, y, z = step_evolution(x, y, z, r, t_step)
#         x_list.append(x)
#         y_list.append(y)
#         z_list.append(z)
#     return x_list, y_list, z_list


# def lyapunov(r, x, y, z, perturbation):
#     sigma, beta = 10, 8 / 3
#     h1 = 0.002
#     jacobian_matrix_raw = np.mat([[-sigma, sigma, 0], [r-z, -1, -x], [y, x, -beta]])
#
#     # normalization
#     perturbation = perturbation / np.linalg.norm(perturbation)
#
#     # fourth-Runge-Kutta Method
#     k1 = np.dot(jacobian_matrix_raw, perturbation)
#     x_new_list, y_new_list, z_new_list = time_evolution(x, y, z, r, time_consume=h1/2)
#     x_new, y_new, z_new = x_new_list[-1], y_new_list[-1], z_new_list[-1]
#     jacobian_matrix = np.mat([[-sigma, sigma, 0], [r - z_new, -1, -x_new], [y_new, x_new, -beta]])
#     perturbation_new = perturbation + h1/2*k1
#
#     k2 = np.dot(jacobian_matrix, perturbation_new)
#     x_new_list, y_new_list, z_new_list = time_evolution(x, y, z, r, time_consume=h1 / 2)
#     x_new, y_new, z_new = x_new_list[-1], y_new_list[-1], z_new_list[-1]
#     jacobian_matrix = np.mat([[-sigma, sigma, 0], [r - z_new, -1, -x_new], [y_new, x_new, -beta]])
#     perturbation_new = perturbation + h1 / 2 * k2
#
#     k3 = np.dot(jacobian_matrix, perturbation_new)
#     x_new_list, y_new_list, z_new_list = time_evolution(x, y, z, r, time_consume=h1)
#     x_new, y_new, z_new = x_new_list[-1], y_new_list[-1], z_new_list[-1]
#     jacobian_matrix = np.mat([[-sigma, sigma, 0], [r - z_new, -1, -x_new], [y_new, x_new, -beta]])
#     perturbation_new = perturbation + h1 * k3
#
#     k4 = np.dot(jacobian_matrix, perturbation_new)
#     perturbation_new = perturbation + h1/6*(k1+2*k2+2*k3+k4)
#
#     delta_perturbation = np.dot(jacobian_matrix, perturbation_new).reshape(-1, 3)
#     lyapunov_exponent = np.dot(delta_perturbation, perturbation_new) / pow(np.linalg.norm(perturbation_new), 2)
#
#     return lyapunov_exponent, perturbation_new, x_new, y_new, z_new


# def r_le_curve(r=28.0, x0=0.1, y0=0, z0=0, length_le=20):
#     x, y, z = x0, y0, z0
#     perturbation = np.random.rand(3, 1)
#
#     last_error = np.inf
#     while True:
#         le_list_single = []
#         for ac_value in range(length_le):
#             le1_single, perturbation, x, y, z = lyapunov(r, x, y, z, perturbation)
#             le_list_single.append(le1_single[0, 0])
#         le1 = sum(le_list_single)/length_le
#         # print('############################')
#         # print('standard deviation: {}'.format(np.std(le_list_single)))
#         # print('lyapunov exponent: {}'.format(le1))
#         # print('lyapunov_list :{}'.format(le_list_single))
#         # print('#############################')
#         if abs(np.std(le_list_single) - last_error) <= 0.01:
#             break
#         else:
#             last_error = np.std(le_list_single)
#
#     return le_list_single, x, y, z, perturbation, le1


if __name__ == '__main__':
    # ac_list = np.linspace(0, 100, 101)
    # results = []
    # for ac_value in ac_list:
    #     result_1 = ac_value*2+1
    #     results.append(result_1)
    # df = {'ac_list': ac_list, 'result': results}
    # df = pd.DataFrame(df)
    # print(df)
    # df.to_excel('result.xlsx')

    # ###############################################################################################################
    # a code to draw figures in nature com
    # df_lyapunov = pd.ExcelFile('E:\\磁\\data\\lyapunov.xlsx')
    # df_lyapunov_1 = pd.read_excel(df_lyapunov, 'periodic')
    # print(df_lyapunov_1['frequency'])
    # x1 = [0]*len(df_lyapunov_1['frequency'][1:])
    # plt.figure()
    # plt.plot(df_lyapunov_1['frequency'][1:], df_lyapunov_1['a'][1:])
    # plt.scatter(df_lyapunov_1['frequency'][1:], df_lyapunov_1['a'][1:], c='red')
    # plt.plot(df_lyapunov_1['frequency'][1:], x1, c='black')
    # plt.xlabel('ac amplitude unit: Oe')
    # plt.ylabel('largest lyapunov exponent')
    # plt.title('largest lyapunov exponent')
    # plt.show()

    # df_delay = pd.ExcelFile('E:\\磁\\data\\Delay.xlsx')
    #
    # df_delay_0 = pd.read_excel(df_delay, 'delay0')
    # df_delay_1 = pd.read_excel(df_delay, 'delay1')
    # df_delay_2 = pd.read_excel(df_delay, 'delay2')
    #
    # df_parity = pd.ExcelFile('E:\\磁\\data\\Parity.xlsx')
    # df_parity_0 = pd.read_excel(df_parity, 'parity0')
    # df_parity_1 = pd.read_excel(df_parity, 'parity1')
    # df_parity_2 = pd.read_excel(df_parity, 'parity2')
    #
    # zero_line = np.linspace(-0.1, 1.1, 20)
    # x_zero_line = [0]*len(zero_line)
    #
    # capacity_delay = np.add(df_delay_0['average'].values, df_delay_1['average'].values)
    # capacity_delay = np.add(capacity_delay, df_delay_2['average'].values)
    # capacity_delay = capacity_delay / max(capacity_delay)
    #
    # capacity_parity = np.add(df_parity_0['average'].values, df_parity_1['average'].values)
    # # capacity_parity = np.add(capacity_parity, df_parity_2['average'].values)
    # capacity_parity = capacity_parity / max(capacity_parity)
    #
    # classification_task = np.add(df_delay_0['average'].values[9:100], df_parity_0['average'].values[9:100])/2
    #
    # plt.figure()
    # plt.plot(x_zero_line, zero_line, c='black', label='zero line')
    # # plt.scatter(df_delay_0['lyapunov'], df_delay_0['average'], label='delay0')
    # # plt.scatter(df_delay_1['lyapunov'], df_delay_1['average'], label='delay1')
    # # plt.scatter(df_delay_2['lyapunov'], df_delay_2['average'], label='delay2')
    # plt.scatter(df_delay_2['lyapunov'], capacity_delay, label='delay')
    # plt.scatter(df_parity_0['lyapunov'], capacity_parity, label='parity')
    # plt.scatter(df_parity_0['lyapunov'][9:100], classification_task, label='classification')
    # # plt.scatter(df_parity_0['lyapunov'], df_parity_0['average'], label='parity0')
    # # plt.scatter(df_parity_1['lyapunov'], df_parity_1['average'], label='parity1')
    # # plt.scatter(df_parity_2['lyapunov'], df_parity_2['average'], label='parity2')
    # plt.xlabel('Lyapunov exponent')
    # plt.ylabel('Memory Capacity / Accuracy')
    # # plt.ylim(-0.1, 1.1)
    # plt.legend()
    #
    # x_major_locator = MultipleLocator(10)
    # ax = plt.gca()
    # # ax.xaxis.set_major_locator(x_major_locator)
    # plt.xlim(-11, 11)
    # plt.show()

    # test for loop time
    # t1 = time.time()
    # j = 0
    # for ac_value in range(1000000):
    #     j += ac_value
    # print(j)
    # t2 = time.time()
    # print('Time : {}s'.format(t2-t1))

    # ######################################################################################################33
    # get data from excel files
    # ac_amplitude_list = np.linspace(60.1, 80.2, 202)
    # path = 'D:\\Python_projects\\Physical-Reservoir-Computing-for-MTJs'
    # le_list = []
    # for ac_amplitude in ac_amplitude_list:
    #     path_new = os.path.join(path, f'lyapunov_random_ac{round(ac_amplitude, 1)}_periodic_f_32000000000.0.xlsx')
    #     df = pd.read_excel(path_new)
    #     print(df['Le'].values[-1])
    #     le_list.append(round(df['Le'].values[-1], 2))
    #
    # data = {'ac': ac_amplitude_list, 'le': le_list}
    # df = pd.DataFrame(data)
    # df.to_excel('result.xlsx')

    # test for interpolation functions in Numpy
    s_in = [1, 0, 1, 0]*10
    real_input = []
    for i in s_in:
        real_input = real_input + [i]*8
    plt.figure()
    plt.plot(real_input)
    plt.title('Input Waveform')
    plt.ylabel(r'Input')
    plt.xlabel(r'Time')
    plt.show()

