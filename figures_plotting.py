import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors

"""
    This code is used to plot useful figures in weekly report and final version of paper.
    Author: Xuezhao WU
    Date: 2022.Sep.16
"""

if __name__ == '__main__':

    # fig
    # plt.figure()
    # consuming_time_list = [1e-9]*20
    # consuming_time_list = [1, 2, 3, 4]
    # plt.xlabel(r'DC current(Oe)', size=16)
    # plt.ylabel(r'$M_{z}$ amplitude (a.u.)', size=16)
    # save_data_path='input_configuration_data/error_bar_random_initial'
    # for i in range(len(consuming_time_list)):
    #     file_name = f'{save_data_path}/input_configuraiton_1.0e-9_{i+21}.xlsx'
    #     raw_data = pd.read_excel(file_name)
    #     data = raw_data.drop(raw_data[raw_data['invalid']==False].index)
    #     plt.plot(data['dc_current'], data['mz_amplitude_max'], label=f'curve {i}')
    
    # # index line
    # x = [90]*20
    # y = np.linspace(-1, 1, 20)
    # plt.plot(x, y, '-', linewidth=5, alpha=0.5)

    # x = [300]*20
    # y = np.linspace(-1, 1, 20)
    # plt.plot(x, y, '-', linewidth=5, alpha=0.5)

    # plt.ylim(0, 0.55)
    # plt.legend()
    # plt.show()

    # ####################################################
    # different ratio of inputs
    # ####################################################

    # posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1)
    # nodes = [16, 20, 30, 40]
    # superposition_list = [1, 2, 3, 4]
    # save_data_path = f'data_stochastic_'
    # # plt.figure()
    # # plt.xlabel(r'$T_{delay}$', size=16)
    # # plt.ylabel(r'$Cor^{2}$', size=16)
    # for node in nodes:
    #     plt.figure()
    #     plt.xlabel(r'$T_{delay}$', size=16)
    #     plt.ylabel(r'$Cor^{2}$', size=16)
    #     plt.title(r'Short Term Memory Task', size=12)
    #     for posibiliy in posibility_list:
    #         data_name = f'data_positibility{posibiliy}_node_{node}_3e-9.csv'
    #         data = pd.read_csv(f'{save_data_path}/{data_name}')
    #         plt.plot(data['superposition'], data['cor_2_delay'], label=f'node:{node} posibility: {posibiliy}')
    #         plt.scatter(data['superposition'], data['cor_2_delay'])
    
    #     plt.legend()
    #     plt.show()

    # for node in nodes:
    #     plt.figure()
    #     plt.xlabel(r'$T_{delay}$', size=16)
    #     plt.ylabel(r'$Cor^{2}$', size=16)
    #     plt.title(r'Parity Check Task', size=12)
    #     for posibiliy in posibility_list:
    #         data_name = f'data_positibility{posibiliy}_node_{node}_3e-9.csv'
    #         data = pd.read_csv(f'{save_data_path}/{data_name}')
    #         plt.plot(data['superposition'], data['cor_2_pc'], label=f'node:{node} posibility: {posibiliy}')
    #         plt.scatter(data['superposition'], data['cor_2_pc'])
    
    #     plt.legend()
    #     plt.show()

    # # ####################################################
    # # different ratio of inputs (test_results)
    # # #################################################### 
    # posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1)
    # nodes = [2, 5, 10, 16, 20, 30, 40, 50, 70, 90, 100]
    # # nodes = [2, 5, 10, 16, 20, 30]
    # superposition_list = np.linspace(1, 10, 10, dtype=int)
    # ac_list = np.linspace(1, 100, 100, dtype=int)
    # task_list = ['Delay', 'Parity']
    # save_data_path = f'test_results/4e-09'
    # for task in task_list:
    #     for ratio in posibility_list:
    #         cor_2_list = []
    #         for node in nodes:
    #             cor_2_sub = 0
    #             for superposition in superposition_list:
    #                 file_name = f'{save_data_path}/{task}{superposition}_node{node}_ratio{ratio}_ac1.csv'
    #                 if not os.path.exists(file_name):
    #                     print(f'no such file: {file_name}')
    #                 data = pd.read_csv(file_name)
    #                 cor_2 = np.mean(data['correlation^2_list'])
    #                 error_2 = np.std(data['correlation^2_list'])
    #                 cor_2_sub += cor_2
    #                 # print(f'delay: {superposition}, cor_2 = {cor_2}, node={node}, ratio={ratio}')
    #             cor_2_list.append(cor_2_sub)
    #         data_frame = pd.DataFrame({'node': nodes, 'cor_2_list': cor_2_list})
    #         data_frame.to_csv(f'test_results/ratio{ratio}_4e-09_{task}.csv')

    # # posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1)
    # # nodes = [2, 5, 10, 16, 20, 30, 40, 50, 70, 90, 100]
    # # superposition_list = np.linspace(1, 10, 10, dtype=int)
    # # ac_list = np.linspace(1, 100, 100, dtype=int)
    # # task_list = ['Delay', 'Parity']
    # # save_data_path = 'test_results'

    # X, Y = np.meshgrid(nodes, posibility_list)

    # cor_list_delay = np.zeros((len(nodes), len(posibility_list)))
    # cor_list_parity = np.zeros((len(nodes), len(posibility_list)))
    # for i in range(len(posibility_list)):
    #     ratio = posibility_list[i]
    #     data_f = pd.read_csv(f'test_results/ratio{ratio}_4e-09_Delay.csv')
    #     cor_list_delay[:, i] = data_f['cor_2_list']
    #     data_f = pd.read_csv(f'test_results/ratio{ratio}_4e-09_Parity.csv')
    #     cor_list_parity[:, i] = data_f['cor_2_list']

    # bins_delay = np.linspace(cor_list_delay.min(), cor_list_delay.max(), 101)
    # nbin_delay = len(bins_delay) - 1
    # norm_delay = mcolors.BoundaryNorm(bins_delay, nbin_delay)
    # cmap_delay = cm.get_cmap('plasma', nbin_delay)
    # # cmap_delay.set_under('white')
    # # cmap_delay.set_over('purple')

    # bins_parity = np.linspace(cor_list_parity.min(), cor_list_parity.max(), 101)
    # nbin_parity = len(bins_parity) - 1
    # norm_parity = mcolors.BoundaryNorm(bins_parity, nbin_parity)
    # cmap_parity = cm.get_cmap('bwr', nbin_parity)
    # # cmap_parity.set_under('white')
    # # cmap_parity.set_over('purple')

    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # x_ticks = np.round(np.linspace(cor_list_delay.min(), cor_list_delay.max(), 10), 2)
    # im = axes[0].contourf(
    #     X, Y, cor_list_delay.T, levels=bins_delay, cmap=cmap_delay, norm=norm_delay, extend='both'
    # )
    # cbar = fig.colorbar(im, ax=axes[0], ticks=x_ticks)
    # axes[0].set_ylabel(r'Ratio', size=16)
    # axes[0].set_xlabel(r'nodes', size=16)
    # axes[0].set_title(r'Delay task', size=16)

    # im = axes[1].contourf(
    #     X, Y, cor_list_parity.T, levels=bins_parity, cmap=cmap_parity, norm=norm_parity, extend='both'
    # )
    # x_ticks = np.round(np.linspace(cor_list_parity.min(), cor_list_parity.max(), 10), 2)
    # cbar = fig.colorbar(im, ax=axes[1], ticks=x_ticks)
    # axes[1].set_title(r'Parity task', size=16)
    # axes[1].set_ylabel(r'Ratio', size=16)
    # axes[1].set_xlabel(r'nodes', size=16)
    # plt.show()

    # plt.subplot(1, 2, 2)
    # for i in range(len(posibility_list)):
    #     plt.scatter(nodes, cor_list_parity[:, i])
    #     plt.plot(nodes, cor_list_parity[:, i], label=f'{posibility_list[i]}')
    #     plt.legend()
    #     plt.xlabel(r'node', size=16)
    #     plt.ylabel(r'$Cor^2$', size=16)
    #     plt.title(r'Parity Check task')

    # plt.subplot(1, 2, 1)
    # for i in range(len(posibility_list)):
    #     plt.scatter(nodes, cor_list_delay[:, i])
    #     plt.plot(nodes, cor_list_delay[:, i], label=f'{posibility_list[i]}')
    #     plt.legend()
    #     plt.xlabel(r'node', size=16)
    #     plt.ylabel(r'$Cor^2$', size=16)
    #     plt.title(r'Delay task')
    # plt.show()

    # for error bar
    # figs
    # for task in task_list:
    #     mc_list = []
    #     error_list = []
    #     for node in nodes:
    #         file_name = f'Best_nodes_{node}_3e-09_{task}.xlsx'
    #         if os.path.exists(file_name):
    #             df = pd.read_excel(file_name)
    #             mc_list.append(np.median(df['sum'].values))
    #             error_list.append(np.std(df['sum'].values)/2)
    #     plt.figure(f'{task}')
    #     plt.ylim(1, 3)
    #     plt.errorbar(x=node_list, y=mc_list, yerr=error_list, label=f'task', fmt='o:', ms=5, capsize=3)
    #     plt.legend(loc='lower right')
    #     plt.xlabel(r'Nodes', fontsize=FontSize)
    #     plt.ylabel(fr'$MC_{task}$', fontsize=FontSize)

    # plt.show()


    # ####################################################
    # different ratio of inputs (test_results)
    # #################################################### 
    # posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1)
    # # nodes = [2, 5, 10, 16, 20, 30]
    # superposition_list = np.linspace(1, 10, 10, dtype=int)
    # ac_list = np.linspace(1, 100, 100, dtype=int)
    # task_list = ['Delay', 'Parity']
    # save_data_path = f'test_results/4e-09'
    # # for task in task_list:
    # #     for ratio in posibility_list:
    # #         cor_2_list = []
    # #         for ac in ac_list:
    # #             cor_2_sub = 0
    # #             for superposition in superposition_list:
    # #                 file_name = f'{save_data_path}/{task}{superposition}_node16_ratio{ratio}_ac{ac}.csv'
    # #                 if not os.path.exists(file_name):
    # #                     print(f'no such file: {file_name}')
    # #                 data = pd.read_csv(file_name)
    # #                 cor_2 = np.mean(data['correlation^2_list'])
    # #                 error_2 = np.std(data['correlation^2_list'])
    # #                 cor_2_sub += cor_2
    # #                 # print(f'delay: {superposition}, cor_2 = {cor_2}, node={node}, ratio={ratio}')
    # #             cor_2_list.append(cor_2_sub)
    # #         data_frame = pd.DataFrame({'ac_current': ac_list, 'cor_2_list': cor_2_list})
    # #         data_frame.to_csv(f'test_results/ratio{ratio}_4e-09_{task}_ac_node16.csv')

    # # posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1)
    # # nodes = [2, 5, 10, 16, 20, 30, 40, 50, 70, 90, 100]
    # # superposition_list = np.linspace(1, 10, 10, dtype=int)
    # # ac_list = np.linspace(1, 100, 100, dtype=int)
    # # task_list = ['Delay', 'Parity']
    # # save_data_path = 'test_results'

    # X, Y = np.meshgrid(ac_list, posibility_list)

    # cor_list_delay = np.zeros((len(ac_list), len(posibility_list)))
    # cor_list_parity = np.zeros((len(ac_list), len(posibility_list)))
    # for i in range(len(posibility_list)):
    #     ratio = posibility_list[i]
    #     data_f = pd.read_csv(f'test_results/ratio{ratio}_4e-09_Delay_ac_node16.csv')
    #     cor_list_delay[:, i] = data_f['cor_2_list']
    #     data_f = pd.read_csv(f'test_results/ratio{ratio}_4e-09_Parity_ac_node16.csv')
    #     cor_list_parity[:, i] = data_f['cor_2_list']

    # bins_delay = np.linspace(cor_list_delay.min(), cor_list_delay.max(), 101)
    # nbin_delay = len(bins_delay) - 1
    # norm_delay = mcolors.BoundaryNorm(bins_delay, nbin_delay)
    # cmap_delay = cm.get_cmap('plasma', nbin_delay)
    # # cmap_delay.set_under('white')
    # # cmap_delay.set_over('purple')

    # bins_parity = np.linspace(cor_list_parity.min(), cor_list_parity.max(), 101)
    # nbin_parity = len(bins_parity) - 1
    # norm_parity = mcolors.BoundaryNorm(bins_parity, nbin_parity)
    # cmap_parity = cm.get_cmap('bwr', nbin_parity)
    # # cmap_parity.set_under('white')
    # # cmap_parity.set_over('purple')

    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # x_ticks = np.round(np.linspace(cor_list_delay.min(), cor_list_delay.max(), 10), 2)
    # im = axes[0].contourf(
    #     X, Y, cor_list_delay.T, levels=bins_delay, cmap=cmap_delay, norm=norm_delay, extend='both'
    # )
    # cbar = fig.colorbar(im, ax=axes[0], ticks=x_ticks)
    # axes[0].set_ylabel(r'Ratio', size=16)
    # axes[0].set_xlabel(r'ac current $a_j$', size=16)
    # axes[0].set_title(r'Delay task', size=16)

    # im = axes[1].contourf(
    #     X, Y, cor_list_parity.T, levels=bins_parity, cmap=cmap_parity, norm=norm_parity, extend='both'
    # )
    # x_ticks = np.round(np.linspace(cor_list_parity.min(), cor_list_parity.max(), 10), 2)
    # cbar = fig.colorbar(im, ax=axes[1], ticks=x_ticks)
    # axes[1].set_title(r'Parity task', size=16)
    # axes[1].set_ylabel(r'Ratio', size=16)
    # axes[1].set_xlabel(r'ac current $a_j$', size=16)
    # plt.show()

    # # obtain the lyapounov exponent 
    # file = f'Data_LLE_trajectory_AC'
    # le_list = np.zeros((len(ac_list), 1))
    # for i in range(len(ac_list)):
    #     ac_value = ac_list[i]
    #     file_name = f'{file}/lyapunov_periodic_ac{ac_value}_nodes16_random.csv'
    #     data_frame = pd.read_csv(file_name)
    #     le_list[i, 0] = data_frame['Le'].tolist()[-1]

    # plt.subplot(1, 2, 2)
    # for i in range(len(posibility_list)):
    #     plt.scatter(le_list[:, 0], cor_list_parity[:, i]+i, label=f'{posibility_list[i]}')
    #     # plt.plot(le_list[:, 0], cor_list_parity[:, i])
    #     plt.legend()
    #     plt.xlabel(r'lyapunov exponent', size=16)
    #     plt.ylabel(r'$Cor^2$', size=16)
    #     plt.xlim([-1, 1])
    #     plt.title(r'Parity Check task')
    # plt.axvline(x=0, alpha=0.5, c='black', linestyle='--')

    # plt.subplot(1, 2, 1)
    # for i in range(len(posibility_list)):
    #     plt.scatter(le_list[:, 0], cor_list_delay[:, i]+i, label=f'{posibility_list[i]}')
    #     # plt.plot(le_list[:, 0], cor_list_delay[:, i])
    #     plt.legend()
    #     plt.xlabel(r'lyapunov exponent', size=16)
    #     plt.ylabel(r'$Cor^2$', size=16)
    #     plt.title(r'Delay task')
    #     plt.xlim([-1, 1])
    # plt.axvline(x=0, alpha=0.5, c='black', linestyle='--')
    # plt.show()

    # plt.figure('lyapunov-ac')
    # plt.plot(ac_list, le_list[:, 0])
    # plt.scatter(ac_list, le_list[:, 0])
    # plt.xlabel(r'ac current $a_j$(Oe)', size=16)
    # plt.ylabel(r'lyapunov exponent $\lambda_i$', size=16)
    # plt.show()

    # ####################################################
    # ipc analyze for second example in paper
    # #################################################### 
    file_path = 'ipc_data'
    sigma_list = np.round(np.linspace(0.2, 2, 10), 1)
    # sigma_list = sigma_list[1:]
    degree_delay_family = [[1, 500], [2, 100], [3, 50], [4, 10]]
    degree_delay_family = [[1, 500]]
    
    ipc_list = []
    for sigma in sigma_list:
        file_name = f'{file_path}/sigma_{sigma}_degree_1_delay_500.csv'
        data_frame = pd.read_csv(file_name)
        ipc_list.append(np.sum(data_frame['c_thresold_list']))

    plt.figure('one-layer reservoir ipc')
    plt.bar(sigma_list, ipc_list, fc='blue', edgecolor='black', width=0.2, tick_label=sigma_list)
    plt.xlabel(r'$\sigma$', size=16)
    plt.ylabel(r'$C_{tot}$', size=16)
    plt.title(r'Legendre', size=16)

    # ipc_list_3 = np.zeros((len(sigma_list), 1))
    # # sigma_list = [1.8, 2]
    # for i in range(len(sigma_list)):
    #     sigma = sigma_list[i]
    #     file_name = f'{file_path}/sigma_{sigma}_degree_3_delay_50.csv'
    #     print(file_name)
    #     if not os.path.exists(file_name):
    #         continue
    #     data_frame = pd.read_csv(file_name)
    #     ipc_list_3[i, 0] = (np.sum(data_frame['c_thresold_list']))
    
    # plt.bar(sigma_list, ipc_list_3[:, 0], fc='green', edgecolor='black', width=0.2, tick_label=sigma_list, bottom=ipc_list)
    plt.show()


