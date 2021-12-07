import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import edge_esn_oscillator as o_mtj
import switch_mtj as s_mtj
from matplotlib.pyplot import MultipleLocator
from scipy.signal import argrelmax
from scipy.interpolate import interp1d


if __name__ == '__main__':
    work_path = 'D:\\Python_projects\\Physical_RC_data'
    if os.path.exists(work_path):
        os.chdir(work_path)
    print(f'Path: {os.getcwd()}')
    # hyper parameters
    FontSize = 15
    LabelSize = 16
    # #################################################################################
    # FIG 1 three different regimes of MTJ and in-set is the bifurcation diagram of chaos
    # ###################################################################################
    # plt.close('all')
    # switching_demo = s_mtj.Mtj(x0=1, y0=0.1, z0=0.01)
    # oscillating_demo = o_mtj.Mtj()
    # chaos_demo = o_mtj.Mtj()
    # mx_list, _, _, t_s = switching_demo.time_evolution(dc_amplitude=200, time_consumed=5e-9)
    # _, _, mz_list, t_o = oscillating_demo.time_evolution(dc_amplitude=200, time_consumed=5e-8)
    # _, _, mz_chaos, t_c = chaos_demo.time_evolution(ac_amplitude=26, dc_amplitude=260, time_consumed=8e-8)
    #
    # plt.figure('FIG 1', figsize=(10, 4.2))
    # plt.subplot(1, 3, 1)
    # plt.text(s='a', x=-0.5, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.title('Switching', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.plot(t_s*1e9, mx_list, c='green', label='switching')
    # plt.xticks(fontproperties='Times New Roman')
    # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.ylabel(r'$M_z$', fontsize=FontSize)
    # plt.yticks([-1, 0, 1])
    # plt.ylim(-1.1, 1.1)
    #
    # plt.subplot(1, 3, 2)
    # plt.text(s='b', x=23, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.title("Oscillating", fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.plot(t_o[int(len(mz_list) / 2):]*1e9, mz_list[int(len(mz_list) / 2):], c='orange')
    # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.xticks(fontproperties='Times New Roman')
    # plt.ylim(-1.1, 1.1)
    # plt.yticks([-1, 0, 1])
    #
    # plt.subplot(1, 3, 3)
    # plt.text(s='c', x=38, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.title('Chaos', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.plot(t_c[int(len(mz_chaos) / 2):]*1e9, mz_chaos[int(len(mz_chaos) / 2):], c='blue')
    # plt.xticks(fontproperties='Times New Roman')
    # plt.ylim(-1.1, 1.1)
    # plt.yticks([-1, 0, 1])
    #
    # save_path = os.path.join(os.getcwd(), 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_1.png', dpi=1200)
    # plt.show()

    # #################################################################################
    # FIG 2 three different behaviors for different lyapunov exponents
    # ###################################################################################
    # plt.close('all')
    # negative_lyapunov_demo = o_mtj.Mtj(x0=1.0, y0=1.0, z0=0.1)
    # zero_lyapunov_demo = o_mtj.Mtj(x0=1.0, y0=1.0, z0=0.1)
    # positive_lyapunov_demo = o_mtj.Mtj(x0=1.0, y0=1.0, z0=0.1)
    #
    # _, my_n, mz_negative, t_n = negative_lyapunov_demo.time_evolution(dc_amplitude=200, time_consumed=3e-8)
    # _, my_e, mz_edge, t_e = zero_lyapunov_demo.time_evolution(dc_amplitude=269, ac_amplitude=26, time_consumed=3e-8)
    # _, my_c, mz_chaos, t_p = positive_lyapunov_demo.time_evolution(ac_amplitude=26, dc_amplitude=260,
    #                                                                time_consumed=3e-8)
    #
    # plt.figure('FIG 2', figsize=(10, 4.2))
    # plt.subplot(1, 3, 1)
    # plt.text(s='(a)', x=14, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.title(r'$\lambda < 0$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.plot(t_n[int(len(t_n)/2):] * 1e9, mz_negative[int(len(t_n)/2):], c='orange', label='ordered')
    # plt.xticks(fontproperties='Times New Roman')
    # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.ylabel(r'$M_z$', fontsize=FontSize)
    # plt.ylim(-0.6, 1.1)
    # plt.yticks([-0.5, 0, 0.5, 1])
    #
    # fig = plt.subplot(1, 3, 2)
    # plt.text(s='(b)', x=14, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.title(r'$\lambda \approx 0$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.plot(t_e[int(len(t_e)/2):] * 1e9, mz_edge[int(len(t_e)/2):], c='red')
    # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.xticks(fontproperties='Times New Roman')
    # plt.ylim(-0.6, 1.1)
    # plt.yticks([-0.5, 0, 0.5, 1])
    #
    # plt.subplot(1, 3, 3)
    # plt.text(s='(c)', x=14, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.title(r'$\lambda > 0$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.plot(t_p[int(len(t_p)/2):] * 1e9, mz_chaos[int(len(t_p)/2):], c='blue')
    # plt.xticks(fontproperties='Times New Roman')
    # plt.ylim(-0.6, 1.1)
    # plt.yticks([-0.5, 0, 0.5, 1])
    #
    # save_path = os.path.join(os.getcwd(), 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_2.png', dpi=1200)
    # plt.show()
    #
    # plt.figure('in_set of FIG2')
    # plt.ylabel(r'$m_z$', fontsize=FontSize)
    # plt.xlabel(r'$m_y$', fontsize=FontSize)
    # plt.plot(my_e[2000:], mz_edge[2000:], c='red')
    # plt.xticks(fontproperties='Times New Roman')
    # save_path = os.path.join(os.getcwd(), 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_2_inset01.png', dpi=1200)
    # plt.show()
    #
    # plt.figure('in_set of FIG2_02')
    # plt.ylabel(r'$m_z$', fontsize=FontSize)
    # plt.xlabel(r'$m_y$', fontsize=FontSize)
    # plt.plot(my_c[2000:], mz_chaos[2000:], c='blue')
    # plt.xticks(fontproperties='Times New Roman')
    # save_path = os.path.join(os.getcwd(), 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_2_inset02.png', dpi=1200)
    #
    # plt.figure('in_set of FIG2_03')
    # plt.ylabel(r'$m_z$', fontsize=FontSize)
    # plt.xlabel(r'$m_y$', fontsize=FontSize)
    # plt.plot(my_n[int(len(my_n)/2):], mz_negative[int(len(my_n)/2):], c='orange')
    # plt.xticks(fontproperties='Times New Roman')
    # save_path = os.path.join(os.getcwd(), 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_2_inset03.png', dpi=1200)
    # plt.show()

    # #################################################################################
    # FIG 3 two corresponding method to build echo state network
    # ###################################################################################
    # switch_device = s_mtj.Mtj(x0=1, y0=0.01, z0=0.01)
    # s_in = np.random.randint(0, 2, 20)
    # mx_whole = []
    # dc_list = []
    # time_consume = 8e-9
    # for i in s_in:
    #     if i == 1:
    #         dc_amplitude = 200
    #     else:
    #         dc_amplitude = -200
    #     dc_list.append(dc_amplitude)
    #     mx_list, _, _, t_list = switch_device.time_evolution(dc_amplitude=dc_amplitude, time_consumed=time_consume)
    #     # if len(mx_whole) != 0:
    #     #     mx_whole = np.append(mx_whole, mx_whole[-1])
    #     mx_whole = np.append(mx_whole, mx_list)
    #     # mx_whole = np.append(mx_whole, mx_list[-1])
    #     # t_whole = np.append(t_whole, t_list)
    #
    #     if len(dc_list) == 1:
    #         # sampling points
    #         length_mx = len(mx_list)
    #         nodes_stm = 10
    #         index_sample = np.linspace(0, len(mx_list) - 1, nodes_stm, dtype=int, endpoint=False)
    #         x_matrix1 = np.array([mx_list[index_value] for index_value in index_sample])
    #         t_matrix1 = np.array([t_list[index_value] for index_value in index_sample])
    # # plt.figure('Process of switch ESN')
    # plt.figure('FIG3 01 switch ESN', figsize=[6, 14.5])
    # plt.subplot(4, 1, 1)
    # plt.title(r'Switching MTJ - Echo State Network', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.text(s='(a)', x=-14, y=1.4, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # # plt.title(r'$\lambda < 0$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # t_in = np.linspace(0, len(s_in)-1, len(s_in))*time_consume
    # plt.plot(t_in * 1e9, s_in, c='blue', label='inputs', marker='v')
    # x_ticks = t_in*1e9
    # # x_ticks = [i+4 for i in x_ticks]
    # plt.xticks(x_ticks, fontproperties='Times New Roman')
    # plt.grid(axis='x')
    # # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.ylabel(r'$S_in$', fontsize=FontSize)
    # plt.ylim(-0.1, 1.1)
    # plt.yticks([0, 0.5, 1])
    #
    # plt.subplot(4, 1, 2)
    # # plt.text(s='(b)', x=14, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # # plt.title(r'$\lambda \approx 0$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # dc_real = []
    # t_dc = []
    # for i in range(len(dc_list)):
    #     if s_in[i] != s_in[i-1] and i != 0:
    #         dc_real.append(dc_list[i-1])
    #         t_dc.append(t_in[i]*1e9)
    #
    #     dc_real.append(dc_list[i])
    #     t_dc.append(t_in[i]*1e9)
    # # t_in = np.linspace(0, len(s_in) - 1, len(dc_real)) * 8e-9
    # plt.plot(t_dc, dc_real, c='blue', label='real inputs')
    # plt.grid(axis='x')
    # # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.ylabel(r'$H_{in}$ / Oe', fontsize=FontSize)
    # # x_ticks = np.linspace(0, 80, 11)
    # # x_ticks = [i + 4 for i in x_ticks]
    # plt.xticks(x_ticks, fontproperties='Times New Roman')
    # # plt.ylim(-0.6, 1.1)
    # plt.yticks([-200, 0, 200])
    #
    # plt.subplot(4, 1, 3)
    # # plt.text(s='(c)', x=14, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # # plt.title(r'$\lambda > 0$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # t_whole = np.linspace(0, len(s_in), len(mx_whole))*time_consume
    # plt.grid(axis='x')
    # plt.ylabel(r'$M_z$', fontsize=FontSize)
    # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.plot(t_whole[:-length_mx] * 1e9, mx_whole[:-length_mx], c='red')
    # # x_ticks = np.linspace(0, 80, 11)
    # plt.xticks(x_ticks, fontproperties='Times New Roman')
    # plt.ylim(-1.1, 1.1)
    # plt.yticks([-1, -0.5, 0, 0.5, 1])
    #
    # # plt.figure('FIG3 01 switch ESN')
    # plt.subplot(4, 1, 4)
    # # plt.text(s='(c)', x=14, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.ylabel(r'$M_z$', fontsize=FontSize)
    # plt.plot(t_whole[0:length_mx] * 1e9, mx_whole[0:length_mx], c='red')
    # plt.scatter(t_matrix1 * 1e9, x_matrix1, c='black')
    # plt.xticks(fontproperties='Times New Roman')
    # plt.ylim(-1.1, 1.1)
    # plt.yticks([-1, 0, 1])
    # save_path = os.path.join(os.getcwd(), 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_3_01.png', dpi=1200)
    # plt.show()

    # ########################################################################################################
    # for oscillator
    # oscillator_device = o_mtj.Mtj(x0=1, y0=0.01, z0=0.01)
    # s_in = np.random.randint(0, 2, 20)
    # mx_whole = []
    # dc_list = []
    # time_consume = 8e-9
    # for i in s_in:
    #     if i == 1:
    #         dc_amplitude = 200
    #     else:
    #         dc_amplitude = 100
    #     dc_list.append(dc_amplitude)
    #     _, _, mx_list, t_list = oscillator_device.time_evolution(dc_amplitude=dc_amplitude,
    #                                                              time_consumed=time_consume)
    #     # if len(mx_whole) != 0:
    #     #     mx_whole = np.append(mx_whole, mx_whole[-1])
    #     mx_whole = np.append(mx_whole, mx_list)
    #     # mx_whole = np.append(mx_whole, mx_list[-1])
    #     # t_whole = np.append(t_whole, t_list)
    #
    #     if len(dc_list) == 2:
    #         # sampling points
    #         length_mx = len(mx_list)
    #         nodes_stm = 10
    #         mz_list_all = mx_list[argrelmax(mx_list)]
    #         t_list_all = t_list[argrelmax(mx_list)]
    #         xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
    #         fp = mz_list_all
    #         sampling_x_values = np.linspace(1, len(mz_list_all), nodes_stm)
    #         # linear slinear quadratic cubic
    #         f = interp1d(xp, fp, kind='quadratic')
    #         mz_sampling = f(sampling_x_values)
    # # plt.figure('Process of switch ESN')
    # plt.figure('FIG3 02 oscillator ESN', figsize=[6, 14.5])
    # plt.subplot(4, 1, 1)
    # plt.title(r'Oscillating MTJ - Echo State Network', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.text(s='(b)', x=-14, y=1.4, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # # plt.title(r'$\lambda < 0$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # t_in = np.linspace(0, len(s_in) - 1, len(s_in)) * time_consume
    # plt.plot(t_in * 1e9, s_in, c='blue', label='inputs', marker='v')
    # x_ticks = t_in * 1e9
    # # x_ticks = [i+4 for i in x_ticks]
    # plt.xticks(x_ticks, fontproperties='Times New Roman')
    # plt.grid(axis='x')
    # # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.ylabel(r'$S_in$', fontsize=FontSize)
    # plt.ylim(-0.1, 1.1)
    # plt.yticks([0, 0.5, 1])
    #
    # plt.subplot(4, 1, 2)
    # # plt.text(s='(b)', x=14, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # # plt.title(r'$\lambda \approx 0$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # dc_real = []
    # t_dc = []
    # for i in range(len(dc_list)):
    #     if s_in[i] != s_in[i - 1] and i != 0:
    #         dc_real.append(dc_list[i - 1])
    #         t_dc.append(t_in[i] * 1e9)
    #
    #     dc_real.append(dc_list[i])
    #     t_dc.append(t_in[i] * 1e9)
    # # t_in = np.linspace(0, len(s_in) - 1, len(dc_real)) * 8e-9
    # plt.plot(t_dc, dc_real, c='blue', label='real inputs')
    # plt.grid(axis='x')
    # # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.ylabel(r'$H_{in}$ / Oe', fontsize=FontSize)
    # # x_ticks = np.linspace(0, 80, 11)
    # # x_ticks = [i + 4 for i in x_ticks]
    # plt.xticks(x_ticks, fontproperties='Times New Roman')
    # # plt.ylim(-0.6, 1.1)
    # plt.yticks([100, 200])
    #
    # plt.subplot(4, 1, 3)
    # # plt.text(s='(c)', x=14, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # # plt.title(r'$\lambda > 0$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # t_whole = np.linspace(0, len(s_in), len(mx_whole)) * time_consume
    # plt.grid(axis='x')
    # plt.ylabel(r'$M_z$', fontsize=FontSize)
    # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.plot(t_whole[int(length_mx*3):length_mx*7] * 1e9, mx_whole[int(length_mx*3):length_mx*7], c='red')
    # # x_ticks = np.linspace(0, 80, 11)
    # plt.xticks(x_ticks[3:8], fontproperties='Times New Roman')
    # # plt.ylim(-1.1, 1.1)
    # plt.yticks([-0.5, 0, 0.5])
    #
    # # plt.figure('FIG3 01 switch ESN')
    # plt.subplot(4, 1, 4)
    # # plt.text(s='(c)', x=14, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.ylabel(r'$M_z$', fontsize=FontSize)
    # plt.plot(t_whole[length_mx*3:length_mx*4] * 1e9, mx_whole[length_mx*3:length_mx*4], c='red')
    # length_mx = len(mx_list)
    # nodes_stm = 10
    # mx_list = np.array(mx_whole[length_mx*3:length_mx*4])
    # t_list = t_whole[length_mx*3:length_mx*4]
    # mz_list_all = mx_list[argrelmax(mx_list)]
    # t_list_all = t_list[argrelmax(mx_list)]
    # mz_list_all = mz_list_all[np.arange(0, len(mz_list_all), 2)]
    # t_list_all = t_list_all[np.arange(0, len(t_list_all), 2)]
    # print(t_list_all)
    # print(mz_list_all)
    # plt.scatter(t_list_all * 1e9, mz_list_all, c='black')
    # plt.xticks(fontproperties='Times New Roman')
    # # plt.ylim(-1.1, 1.1)
    # plt.yticks([-0.5, 0, 0.5])
    # # save_path = os.path.join(os.getcwd(), 'FIGURES_PAPER')
    # # if not os.path.exists(save_path):
    # #     os.mkdir(save_path)
    # # plt.savefig(f'{save_path}/FIG_3_02.png', dpi=1200)
    # plt.show()
    #
    # # output
    # s_in = np.random.randint(0, 2, 20)
    # plt.figure('FIG3 02_2 oscillator ESN')
    # # plt.title(r'Oscillating MTJ - Echo State Network', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.text(s='(c)', x=-8, y=1.21, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # # plt.title(r'$\lambda < 0$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # t_in = np.linspace(0, len(s_in) - 1, len(s_in)) * time_consume
    # plt.plot(t_in * 1e9, s_in, c='red', label='inputs', marker='^')
    # x_ticks = t_in * 1e9
    # # x_ticks = [i+4 for i in x_ticks]
    # plt.xticks(x_ticks, fontproperties='Times New Roman')
    # plt.grid(axis='x')
    # # plt.xlabel(r'T(ns)', fontsize=FontSize)
    # plt.ylabel(r'$y_{out}$', fontsize=FontSize)
    # plt.ylim(-0.1, 1.1)
    # plt.yticks([0, 0.5, 1])
    # plt.xlabel(r'$T(ns)$', fontsize=FontSize)
    # save_path = os.path.join(os.getcwd(), 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_3_03.png', dpi=1200)
    # plt.show()

    # ########################################################################################################
    # for best size and evolution time
    # ########################################################################################################
    # df_parity = pd.read_excel('E:\\磁\\data\\Best_size_Parity.xlsx')
    # df_delay = pd.read_excel('E:\\磁\\data\\Best_size_delay.xlsx')
    #
    # plt.figure('Best reservoir size for delay task')
    # plt.subplot(2, 1, 1)
    # # plt.xlabel(r'Nodes', fontsize=FontSize)
    # plt.text(s='(b)', x=-5, y=3.4, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.title(r'Oscillating ESN', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.ylabel(r'$MC_{STM}$', fontsize=FontSize)
    # plt.plot(df_delay['node'], df_delay['sum'], c='blue', label='Delay')
    # plt.scatter(df_delay['node'], df_delay['sum'], c='blue')
    # plt.ylim(0, 3)
    # plt.legend(loc='lower right')
    #
    # # plt.figure('Best reservoir size for parity task')
    # # plt.title('Best reservoir size')
    # plt.subplot(2, 1, 2)
    # plt.xlabel(r'Nodes', fontsize=FontSize)
    # plt.ylabel(r'$MC_{PC}$', fontsize=FontSize)
    # error_range = [0.1] * len(df_parity['node'])
    # # plt.errorbar(df_parity['node'], df_parity['SUM'], c='red', label='parity', fmt='o:', ecolor='hotpink',
    # # elinew1idth=3,
    # #              ms=5, mfc='wheat', mec='salmon',yerr=error_range,
    # #              capsize=3)
    # plt.plot(df_parity['node'], df_parity['SUM'], c='red', label='Parity')
    # plt.scatter(df_parity['node'], df_parity['SUM'], c='red')
    # plt.yticks([1.0, 1.5, 2.0, 2.5])
    # plt.legend(loc='lower right')
    # save_path = os.path.join(os.getcwd(), 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_4b.png', dpi=1200)
    # # plt.show()
    #
    # # for switch MTJ
    # os.chdir('D:\\Python_projects\\Physical_RC_data')
    # df_delay = pd.read_excel('best_node_switch_delay_8e-9.xlsx')
    # df_parity = pd.read_excel('best_node_switch_Parity_8e-9.xlsx')
    #
    # plt.figure('Best reservoir size')
    # plt.subplot(2, 1, 1)
    # # plt.xlabel(r'Nodes', fontsize=FontSize)
    # plt.text(s='(a)', x=-5, y=3.4, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.title(r'Switching ESN', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.ylabel(r'$MC_{STM}$', fontsize=FontSize)
    # plt.errorbar(df_delay['node'], df_delay['average'], c='blue', label='Delay', fmt='o:', ecolor='blue',
    #              ms=5, yerr=df_delay['error'],
    #              capsize=3)
    # # plt.plot(df_delay['node'], df_delay['average'], c='blue', label='Delay')
    # # plt.scatter(df_delay['node'], df_delay['average'], c='blue')
    # plt.ylim(0, 3)
    # plt.legend(loc='lower right')
    #
    # # plt.figure('Best reservoir size for parity task')
    # # plt.title('Best reservoir size')
    # plt.subplot(2, 1, 2)
    # plt.xlabel(r'Nodes', fontsize=FontSize)
    # plt.ylabel(r'$MC_{PC}$', fontsize=FontSize)
    # plt.errorbar(df_parity['node'], df_parity['average'], c='red', label='parity', fmt='o:', ecolor='red',
    #              ms=5, yerr=df_parity['error'],
    #              capsize=3)
    # # plt.plot(df_parity['node'], df_parity['average'], c='red', label='Parity')
    # # plt.scatter(df_parity['node'], df_parity['average'], c='red')
    # plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3])
    # plt.legend(loc='lower right')
    # save_path = os.path.join(os.getcwd(), 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_4a.png', dpi=1200)
    # plt.show()

    # ########################################################################################################
    # for best evolution time
    # ########################################################################################################
    # os.chdir('D:\\Python_projects\\Physical_RC_data\\best_time')
    # time_list = [2e-9, 3e-9, 4e-9, 6e-9, 7e-9, 10e-9, 20e-9]
    # task_list = ['Delay', 'Parity']
    # number = np.linspace(0, 9, 10, dtype=int)
    # # print(number)
    # node_list = [10, 20, 30]
    #
    # plt.figure('Best time for oscillator esn for delay task', figsize=[5, 6])
    # plt.subplot(2, 1, 1)
    # plt.title(r'Oscillating ESN', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.text(s='(a)', x=-1, y=3.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # # plt.xlabel(r'Evolution Time(ns)', fontsize=FontSize)
    # plt.ylabel(r'$MC_{STM}$', fontsize=FontSize)
    # plt.ylim(0, 3)
    # time_list_normal = [i * 1e9 for i in time_list]
    # plt.xticks(time_list_normal)
    # task = 'Delay'
    # for node in node_list:
    #     final_value_list, error_list = [], []
    #     for time_value in time_list:
    #         file_path = f'Best_time_node_{node}_{task}_{time_value}.xlsx'
    #         if os.path.exists(file_path):
    #             df = pd.read_excel(file_path)
    #             max_df = np.max(df.iloc[0, 1:]) + np.max(df.iloc[1, 1:]) + np.max(df.iloc[2, 1:]) + np.max(
    #                 df.iloc[3, 1:]) + 1
    #             min_df = np.min(df.iloc[0, 1:]) + np.min(df.iloc[1, 1:]) + np.min(df.iloc[2, 1:]) + np.min(
    #                 df.iloc[3, 1:]) + 1
    #             error_df = (max_df - min_df) / 2 / ((max_df + min_df) / 2)
    #             final_value_list.append((max_df + min_df) / 2)
    #             error_list.append(error_df)
    #         else:
    #             print('no file')
    #     time_list_normal = [i * 1e9 for i in time_list]
    #     plt.errorbar(time_list_normal, final_value_list, yerr=error_list, fmt='o:',
    #                  capsize=2,
    #                  elinewidth=1, label=f' size {node}')
    #
    # plt.legend(loc='lower right')
    #
    # plt.subplot(2, 1, 2)
    # plt.xlabel(r'Evolution Time(ns)', fontsize=FontSize)
    # plt.ylabel(r'$MC_{PC}$', fontsize=FontSize)
    # plt.ylim(0, 3)
    # time_list_normal = [i * 1e9 for i in time_list]
    # plt.xticks(time_list_normal)
    # task = 'Parity'
    # for node in node_list:
    #     final_value_list, error_list = [], []
    #     for time_value in time_list:
    #         file_path = f'Best_time_node_{node}_{task}_{time_value}.xlsx'
    #         if os.path.exists(file_path):
    #             df = pd.read_excel(file_path)
    #             max_df = np.max(df.iloc[0, 1:]) + np.max(df.iloc[1, 1:]) + np.max(df.iloc[2, 1:]) + np.max(
    #                 df.iloc[3, 1:]) + 1
    #             min_df = np.min(df.iloc[0, 1:]) + np.min(df.iloc[1, 1:]) + np.min(df.iloc[2, 1:]) + np.min(
    #                 df.iloc[3, 1:]) + 1
    #             error_df = (max_df - min_df) / 2 / ((max_df + min_df) / 2)
    #             final_value_list.append((max_df + min_df) / 2)
    #             error_list.append(error_df)
    #         else:
    #             print('no file')
    #     time_list_normal = [i * 1e9 for i in time_list]
    #     plt.errorbar(time_list_normal, final_value_list, yerr=error_list, fmt='o:',
    #                  capsize=2,
    #                  elinewidth=1, label=f' size {node}')
    #
    # plt.legend(loc='lower right')
    #
    # save_path = os.path.join('D:\\Python_projects\\Physical_RC_data', 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_5b.png', dpi=1200)
    # plt.show()

    # for switch
    # node_list = [10, 20, 30, 40, 50]
    #
    # plt.figure('Best time for oscillator esn for delay task', figsize=[5, 6])
    # plt.subplot(2, 1, 1)
    # plt.title(r'Switching ESN', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.text(s='(b)', x=-6, y=3.7, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.ylabel(r'$MC_{STM}$', fontsize=FontSize)
    # plt.ylim(0, 3.5)
    #
    # time_list = [2e-9, 5e-9, 9e-9, 10e-9, 20e-9, 50e-9]
    # time_list_normal = [i * 1e9 for i in time_list]
    # time_list_x = [2e-9, 5e-9, 10e-9, 20e-9, 50e-9]
    # time_list_normal_x = [i * 1e9 for i in time_list_x]
    # plt.xticks(time_list_normal_x)
    # error_range_10 = [0.060588628, 0.08, 0.11, 0.11, 0.070541756, 0.014]
    # y_node_10 = [0.611462252, 1.365214413, 2.72, 2.715089322, 2.195038759, 2.107310922]
    #
    # error_range_20 = [0.09, 0.08, 0.3, 0.2, 0.0490541756, 0.004]
    # y_node_20 = [0.569979257, 1.340590888, 2.488990649, 3.040506322, 2.220101061, 2.130594105]
    #
    # error_range_30 = [0.1, 0.035, 0.31, 0.18, 0.06, 0.02]
    # y_node_30 = [0.528785401, 0.757664606, 2.419943606, 2.419943606, 2.212622348, 2.107310922]
    # plt.errorbar(time_list_normal, y_node_10, yerr=error_range_10, fmt='o:',
    #              capsize=2,
    #              elinewidth=1, label=' size 10')
    # plt.errorbar(time_list_normal, y_node_20, yerr=error_range_20, fmt='o:',
    #              capsize=2,
    #              elinewidth=1, label=' size 20')
    # plt.errorbar(time_list_normal, y_node_30, yerr=error_range_30, fmt='o:',
    #              capsize=2,
    #              elinewidth=1, label=' size 30')
    # plt.legend(loc='lower right')
    # plt.subplot(2, 1, 2)
    # plt.xlabel(r'Evolution Time(ns)', fontsize=FontSize)
    # plt.ylabel(r'$MC_{PC}$', fontsize=FontSize)
    # plt.ylim(0, 3.5)
    # time_list_normal = [i * 1e9 for i in time_list]
    # plt.xticks(time_list_normal_x)
    # error_range_10 = [0.060588628, 0.108, 0.11, 0.228943984, 0.06, 0.065]
    # y_node_10 = [0.256121433, 0.658866073, 1.982309304, 2.467288916, 1.986621256, 1.328295956]
    #
    # error_range_20 = [0.09, 0.16, 0.1, 0.22, 0.0490541756, 0.05]
    # y_node_20 = [0.311197846, 0.319727497, 2.302620206, 3.010506322, 2.178768459, 1.942899982]
    #
    # error_range_30 = [0.1, 0.118012786, 0.5, 0.18, 0.06, 0.044]
    # y_node_30 = [0.317745624, 0.219923186, 2.139171018, 3.179592104, 2.183057185, 2.09393611]
    # plt.errorbar(time_list_normal, y_node_10, yerr=error_range_10, fmt='o:',
    #              capsize=2,
    #              elinewidth=1, label=' size 10')
    # plt.errorbar(time_list_normal, y_node_20, yerr=error_range_20, fmt='o:',
    #              capsize=2,
    #              elinewidth=1, label=' size 20')
    # plt.errorbar(time_list_normal, y_node_30, yerr=error_range_30, fmt='o:',
    #              capsize=2,
    #              elinewidth=1, label=' size 30')
    # plt.legend(loc='lower right')
    # save_path = os.path.join('D:\\Python_projects\\Physical_RC_data', 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_20b.png', dpi=1200)
    # plt.show()

    # ########################################################################################################
    # Parity check task and Delay task concept showing
    # ########################################################################################################
    # oscillator_device = o_mtj.Mtj()
    # oscillator_device.real_time_test(test_number=60, nodes_stm=16, visual_index=True, superposition=1,
    #                                  time_consume_all=4e-9, task='Parity', file_path='weight_matrix_interpolation')
    # plt.show()

    # ########################################################################################################
    # edge of chaos
    # ########################################################################################################
    os.chdir('E:\\磁\\data')
    df_delay = pd.ExcelFile('E:\\磁\\data\\Delay.xlsx')
    df_delay_0 = pd.read_excel(df_delay, 'delay0')
    df_delay_1 = pd.read_excel('delay1_from_0_100.xlsx')
    df_delay_2 = pd.read_excel('delay2.xlsx')

    df_parity = pd.ExcelFile('E:\\磁\\data\\Parity.xlsx')
    df_parity_1 = pd.read_excel('parity1_from_0_100.xlsx')
    df_parity_0 = pd.read_excel(df_parity, 'parity0')
    df_parity_2 = pd.read_excel('parity2.xlsx')

    # zero line drawing
    zero_line = np.linspace(-0.1, 1.1, 20)
    x_zero_line = [0] * len(zero_line)

    plt.figure('delay task - lyapunov', figsize=[6, 8])

    plt.subplot(2, 2, 1)
    plt.text(s='(a)', x=-4, y=1.08, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    plt.title(r'$T_{delay} = 1$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    plt.plot(x_zero_line, zero_line, c='black', label='zero line', linestyle='--')
    plt.scatter(df_delay_1['le'], df_delay_1['capacity'], label='delay1', c='orange', s=10)
    plt.ylabel(r'$C_{STM}$', fontsize=FontSize)
    # plt.xlabel(r'$\lambda$', fontsize=FontSize)
    # plt.ylabel(r'$C_{STM}$', fontsize=FontSize)
    plt.ylim(0.8, 1.1)
    plt.yticks([0.8, 0.9, 1.0])
    # plt.legend()
    x_major_locator = plt.MultipleLocator(4)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-4, 4)
    # plt.xlim(-11, 11)

    plt.subplot(2, 2, 2)
    plt.title(r'$T_{delay} = 2$', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    plt.plot(x_zero_line, zero_line, c='black', label='zero line', linestyle='--')
    plt.scatter(df_delay_2['le'], df_delay_2['average'], label='delay2', c='green', s=10)
    # plt.xlabel(r'$\lambda$', fontsize=FontSize)
    # plt.ylabel(r'$C_{STM}$', fontsize=FontSize)
    plt.text(s='(b)', x=-8, y=0.55, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    plt.ylim(0, 0.6)
    # plt.legend()
    x_major_locator = plt.MultipleLocator(4)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-8, 8)
    # plt.ylim(0, 1)

    plt.subplot(2, 2, 3)
    plt.plot(x_zero_line, zero_line, c='black', label='zero line')
    plt.scatter(df_parity_1['le'], df_parity_1['capacity'], label='parity1', c='orange', s=10)
    plt.xlabel(r'$\lambda$', fontsize=FontSize)
    plt.text(s='(c)', x=-4, y=1.07, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    plt.xlabel(r'$\lambda$', fontsize=FontSize)
    plt.ylabel(r'$C_{PC}$', fontsize=FontSize)
    # plt.ylabel(r'$C_{STM}$', fontsize=FontSize)
    plt.ylim(0.7, 1.1)
    plt.yticks([0.7, 0.8, 0.9, 1.0])
    # plt.legend()
    x_major_locator = plt.MultipleLocator(4)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-4, 4)

    plt.subplot(2, 2, 4)
    plt.plot(x_zero_line, zero_line, c='black', label='zero line')
    plt.scatter(df_parity_2['le'], df_parity_2['average'], label='parity2', c='green', s=10)
    plt.xlabel(r'$\lambda$', fontsize=FontSize)
    # plt.ylabel('Capacity')
    plt.text(s='(d)', x=-8, y=0.55, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    plt.ylim(0, 0.6)
    x_major_locator = plt.MultipleLocator(4)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-8, 8)
    save_path = os.path.join('D:\\Python_projects\\Physical_RC_data', 'FIGURES_PAPER')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(f'{save_path}/FIG_6.png', dpi=1200)
    plt.show()

    # ########################################################################################################
    # lyapunov exponents
    # ########################################################################################################
    # os.chdir('E:\\磁\\data')
    # df = pd.read_excel('lyapunov_mag_new.xlsx')
    #
    # # zero line
    # y_zero = [0]*len(df['ac'].values)
    #
    # plt.figure('lyapunov exponent')
    # plt.title('largest lyapunov exponent spectrum', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.xlabel(r'$H_{ac}$ / Oe', fontsize=LabelSize)
    # plt.ylabel(r'lle', fontsize=LabelSize)
    # plt.scatter(df['ac'], df['le'], c='red', label='lyapunov', s=8)
    # plt.plot(df['ac'], y_zero, c='black', label='zero line', linestyle='--')
    # plt.legend(loc='upper left')
    # save_path = os.path.join('D:\\Python_projects\\Physical_RC_data', 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_7.png', dpi=1200)
    # plt.show()

    # ########################################################################################################
    # comparison between different regime based system
    # ########################################################################################################
    # os.chdir('D:\\Python_projects\\Physical_RC_data\\multi')
    # oscillator_delay = pd.read_excel('Oscillator_node_20_Delay_0.xlsx')
    # oscillator_parity = pd.read_excel('Oscillator_node_20_Parity_0.xlsx')
    #
    # # switch_delay = pd.read_excel('Switch_node_20_Delay_0.xlsx')
    # # switch_parity = pd.read_excel('Switch_node_20_Parity_0.xlsx')
    # error_y_delay = [3.35117E-16,
    #                  2.34056E-16,
    #                  0.041196168,
    #                  0.15149303,
    #                  0.09163594,
    #                  0.05210476,
    #                  0.069922057,
    #                  0.020515063,
    #                  0.044083884,
    #                  0.051667313
    #                  ]
    # switch_y_delay = [1, 1, 0.439154288, 0.218772474, 0.149290475, 0.047029126, 0.039622815, 0.024416077, 0.0353197,
    #                   0.037541648]
    # number_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # plt.figure('comparison')
    # plt.title(r'STM task', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.errorbar(number_list, switch_y_delay, c='green', label='Switching', fmt='o:',
    #              ecolor='green',
    #              ms=5, yerr=error_y_delay, capsize=3)
    # plt.errorbar(oscillator_delay['number'], oscillator_delay['final'], c='orange', label='Oscillating', fmt='o:',
    #              ecolor='orange',
    #              ms=5, yerr=oscillator_delay['error'], capsize=3)
    #
    # plt.xlabel(r'$T_{delay}$', fontsize=LabelSize)
    # plt.ylabel(r'$Cor^2$', fontsize=LabelSize)
    # plt.text(x=4, y=0.8, s='MC :2.04', c='green')
    # plt.text(x=4, y=0.9, s='MC :{:.2f}'.format(np.sum(oscillator_delay["final"].values[1:])), c='orange')
    # plt.legend(loc='lower left')
    # save_path = os.path.join('D:\\Python_projects\\Physical_RC_data', 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_8a.png', dpi=1200)
    #
    # error_y_parity = [2.09346E-16,
    #                   0.060741086,
    #                   0.050642476,
    #                   0.089104763,
    #                   0.096335297,
    #                   0.069085709,
    #                   0.029761267,
    #                   0.03667962,
    #                   0.030410994,
    #                   0.020265142]
    # switch_y_parity = [1,
    #                    0.961271414,
    #                    0.517110804,
    #                    0.199842421,
    #                    0.142021894,
    #                    0.063152113,
    #                    0.029475203,
    #                    0.045262445,
    #                    0.025047135,
    #                    0.01971613
    #                    ]
    # number_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # plt.figure('comparison of Parity')
    # plt.title(r'PC task', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
    # plt.errorbar(number_list, switch_y_parity, c='green', label='Switching', fmt='o:',
    #              ecolor='green',
    #              ms=5, yerr=error_y_parity, capsize=3)
    # plt.errorbar(oscillator_parity['number'], oscillator_parity['final'], c='orange', label='Oscillating', fmt='o:',
    #              ecolor='orange',
    #              ms=5, yerr=oscillator_parity['error'], capsize=3)
    # plt.xlabel(r'$T_{delay}$', fontsize=LabelSize)
    # plt.ylabel(r'$Cor^2$', fontsize=LabelSize)
    # plt.text(x=4, y=0.8, s='MC : 2.01', c='green')
    # plt.text(x=4, y=0.9, s='MC :{:.2f}'.format(np.sum(oscillator_parity["final"].values[1:])), c='orange')
    # plt.legend(loc='lower left')
    # save_path = os.path.join('D:\\Python_projects\\Physical_RC_data', 'FIGURES_PAPER')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # plt.savefig(f'{save_path}/FIG_8b.png', dpi=1200)
    # plt.show()
