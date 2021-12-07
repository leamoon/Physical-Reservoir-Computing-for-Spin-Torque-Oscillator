"""
This code is used to draw all figures and data in Physical reservoir computing project
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import edge_esn_oscillator as o_mtj
import switch_mtj as s_mtj

if __name__ == '__main__':
    # ###############################################################################################
    # information transfer
    # ###############################################################################################
    try:
        path, file_name = 'E:\\磁\\data', 'information_transfer.xlsx'
        current_path = os.path.join(path, file_name)
        df = pd.read_excel(current_path)
        zero_line = np.linspace(0, 1, 10)
        x_axis_zero = [0]*len(zero_line)

        plt.figure('Information Transfer Diagram')
        plt.plot(x_axis_zero, zero_line, c='black', label='zero line')
        plt.scatter(df['LE'], df['capacity'], c='red', label='covariance')
        x_major_locator = plt.MultipleLocator(10)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-11, 11)
        plt.ylim(0, 1)
        plt.title('Information Transfer')
        plt.xlabel(r'lyapunov exponent')
        plt.ylabel(r'Covariance')
        plt.legend()
        # plt.show()
        print(r'Information Transfer part runs successfully !')

    except Exception as ErrorMessage:
        print('error in information transfer')
        # sys.exit(ErrorMessage)

    # ###############################################################################################
    # information Storage
    # ###############################################################################################
    try:
        path, file_name = 'E:\\磁\\data', 'information_storage.xlsx'
        current_path = os.path.join(path, file_name)
        df = pd.read_excel(current_path)
        zero_line = np.linspace(0, 1, 10)
        x_axis_zero = [0] * len(zero_line)

        plt.figure('Information Storage Diagram')
        plt.plot(x_axis_zero, zero_line, c='black', label='zero line')
        plt.scatter(df['LE'], df['capacity'], c='orange', label='covariance')
        x_major_locator = plt.MultipleLocator(10)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-11, 11)
        plt.ylim(0, 1)
        plt.title('Information Storage')
        plt.xlabel(r'lyapunov exponent')
        plt.ylabel(r'Covariance')
        plt.legend()
        # plt.show()
        print(r'Information Storage part runs successfully !')

    except Exception as ErrorMessage:
        print('error in information Storage')
        # sys.exit(ErrorMessage)

    # ###############################################################################################
    # get lyapunov data from multi excel files
    # ###############################################################################################
    try:
        ac_amplitude_list = np.linspace(0, 100, 1001)
        path = 'D:\\Python_projects\\Physical-Reservoir-Computing-for-MTJs'
        le_list, ac_list = [], []
        for ac_amplitude in ac_amplitude_list:
            path_new = os.path.join(path, f'lyapunov_random_ac{round(ac_amplitude, 1)}_periodic_f_32000000000.0.xlsx')
            if os.path.exists(path_new):
                df = pd.read_excel(path_new)
                # print(df['Le'].values[-1])
                ac_list.append(round(ac_amplitude, 1))
                le_list.append(round(df['Le'].values[-1], 2))

        data = {'ac': ac_list, 'le': le_list}
        df = pd.DataFrame(data, columns=['ac', 'le'])
        if os.path.exists('lyapunov_mag.xlsx'):
            df_old = pd.read_excel('lyapunov_mag.xlsx')
            df = pd.concat([df, df_old], ignore_index=True)

        df = df[['ac', 'le']].drop_duplicates()
        df = df.sort_values(by="ac", ignore_index=True)
        df.to_excel('lyapunov_mag.xlsx')
        print(r'Lyapunov collecting from Multi files runs successfully !')

    except Exception as ErrorMessage:
        print('error from Lyapunov collecting from Multi files')
        # sys.exit(ErrorMessage)

    # #################################################################################################
    # draw figures: lyapunov - capacity (delay task and parity task)
    # #################################################################################################
    try:
        os.chdir('E:\\磁\\data')
        df_delay = pd.ExcelFile('E:\\磁\\data\\Delay.xlsx')
        df_delay_0 = pd.read_excel(df_delay, 'delay0')
        df_delay_1 = pd.read_excel('delay1_from_0_100.xlsx')
        df_delay_2 = pd.read_excel('delay2.xlsx')

        df_parity = pd.ExcelFile('E:\\磁\\data\\Parity.xlsx')
        df_parity_1 = pd.read_excel('parity1_from_0_100.xlsx')
        df_parity_0 = pd.read_excel(df_parity, 'parity0')
        df_parity_2 = pd.read_excel('parity2.xlsx')

        # df_lyapunov = pd.read_excel('E:\\磁\\data\\lyapunov_mag.xlsx')
        # # df_lyapunov = pd.read_excel('E:\\磁\\data\\lyapunov_reservoir.xlsx')
        # df_parity_0 = pd.merge(df_lyapunov, df_parity_0)
        # df_parity_1 = pd.merge(df_lyapunov, df_parity_1)
        # df_parity_2 = pd.merge(df_lyapunov, df_parity_2)
        # df_delay_0 = pd.merge(df_lyapunov, df_delay_0)
        # df_delay_1 = pd.merge(df_lyapunov, df_delay_1)
        # df_delay_2 = pd.merge(df_lyapunov, df_delay_2)
        # df = df_parity_2[['ac', 'le', 'capacity']].drop_duplicates()
        # df = df.sort_values(by="ac", ignore_index=True)
        # df.to_excel('E:\\磁\\data\\result.xlsx')
        #
        # zero line drawing
        zero_line = np.linspace(-0.1, 1.1, 20)
        x_zero_line = [0] * len(zero_line)
        #
        # # normalization and calculation of total capacity
        # capacity_delay = np.add(df_delay_0['average'].values, df_delay_1['capacity'].values)
        # capacity_delay = np.add(capacity_delay, df_delay_2['capacity'].values)
        # capacity_delay = capacity_delay / max(capacity_delay)
        #
        # capacity_parity = np.add(df_parity_0['average'].values, df_parity_1['capacity'].values)
        # # capacity_parity = np.add(capacity_parity, df_parity_2['average'].values)
        # capacity_parity = capacity_parity / max(capacity_parity)
        capacity_delay = df_delay_1['capacity'].values + df_delay_2['average'].values
        capacity_delay = capacity_delay / max(capacity_delay)
        capacity_parity = df_parity_1['capacity'].values + df_parity_2['average'].values
        capacity_parity = capacity_parity / max(capacity_parity)

        plt.figure('whole capacity')
        plt.subplot(2, 1, 1)
        plt.plot(x_zero_line, zero_line, c='black', label='zero line', linestyle='--')
        plt.scatter(df_delay_1['le_new'], capacity_delay, label='delay', c='green')
        plt.legend()
        x_major_locator = plt.MultipleLocator(10)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-11, 11)
        plt.ylim(0, 1.1)
        plt.ylabel('Memory Capacity')
        plt.title('Memory capacity of delay task')

        plt.subplot(2, 1, 2)
        plt.plot(x_zero_line, zero_line, c='black', label='zero line', linestyle='--')
        plt.scatter(df_parity_1['le_new'], capacity_parity, label='parity', c='red')
        plt.xlabel('Lyapunov exponent')
        plt.ylabel('Memory Capacity')
        plt.legend()
        x_major_locator = plt.MultipleLocator(10)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-11, 11)
        plt.ylim(0, 1.1)
        plt.title('Memory capacity of parity task')

        plt.figure('delay task - lyapunov')
        plt.title('delay task - lyapunov')
        plt.plot(x_zero_line, zero_line, c='black', label='zero line')
        plt.scatter(df_delay_0['lyapunov_mag'], df_delay_0['average'], label='delay0')
        plt.scatter(df_delay_1['le_new'], df_delay_1['capacity'], label='delay1')
        plt.scatter(df_delay_2['le_new'], df_delay_2['average'], label='delay2')
        plt.xlabel('Lyapunov exponent')
        plt.ylabel('Capacity')
        plt.legend()
        x_major_locator = plt.MultipleLocator(10)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-11, 11)
        # plt.ylim(0, 1)

        plt.figure('Parity task - lyapunov')
        plt.title('Parity task - lyapunov')
        plt.plot(x_zero_line, zero_line, c='black', label='zero line')
        plt.scatter(df_parity_0['lyapunov_mag'], df_parity_0['average'], label='parity0')
        plt.scatter(df_parity_1['le_new'], df_parity_1['capacity'], label='parity1')
        plt.scatter(df_parity_2['le_new'], df_parity_2['average'], label='parity2')
        plt.xlabel('Lyapunov exponent')
        plt.ylabel('Capacity')
        plt.legend()

        x_major_locator = plt.MultipleLocator(10)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-11, 11)
        # plt.ylim(0, 1)
        # plt.show()
        print('lyapunov-capacity figure drawing function runs successfully !')

    except Exception as ErrorMessage:
        print(r'error in lyapunov-capacity figure drawing function !')
        sys.exit(ErrorMessage)

    # #################################################################################################
    # capture time evolution result excel file into a same excel file
    # #################################################################################################
    try:
        index_run = False
        if index_run:
            os.chdir('E:\\磁\\data')
            time_list = [2e-9, 3e-9, 4e-9, 6e-9, 7e-9, 10e-9, 20e-9, 50e-9]
            superposition_list = np.linspace(8, 19, 12, dtype=int)
            task_list = ['Delay']
            for task in task_list:
                for time_value in time_list:
                    file_name = f'Finding_best_time_8_{time_value}_{task}_2021-11-08.xlsx'
                    length_cor = len(pd.read_excel(file_name)['covariance'])
                    if os.path.exists(file_name):
                        data_value = np.zeros((length_cor, int(superposition_list[-1]+1)))
                        for superposition_number in superposition_list:
                            file_path = f'Finding_best_time_{superposition_number}_{time_value}_{task}_2021-11-08.xlsx'
                            data_value[:, superposition_number] = pd.read_excel(file_path)['covariance']
                            # print(file_path)
                    new_data = pd.DataFrame(data_value)
                    # new_data.to_excel(f'{task}_best_time_data_{time_value}.xlsx')
                    new_data.to_excel(f'D:\\Python_projects\\Physical_RC_data\\{task}_best_time_data_{time_value}_2.xlsx')
            print('capture time evolution files runs successfully !')

    except Exception as ErrorMessage:
        print('----Error from capture time evolution file----')
        sys.exit(ErrorMessage)

    # #################################################################################################
    # drawing figures of best reservoir size
    # #################################################################################################
    df_parity = pd.read_excel('E:\\磁\\data\\Best_size_Parity.xlsx')
    df_delay = pd.read_excel('E:\\磁\\data\\Best_size_delay.xlsx')

    plt.figure('Best reservoir size for delay task')
    # plt.title('Best reservoir size')
    plt.xlabel(r'reservoir size')
    plt.ylabel(r'MC')
    plt.plot(df_delay['node'], df_delay['sum'], c='blue', label='Delay')
    plt.scatter(df_delay['node'], df_delay['sum'], c='blue')
    plt.ylim(0, 3)
    plt.legend()

    plt.figure('Best reservoir size for parity task')
    plt.title('Best reservoir size')
    plt.xlabel(r'reservoir size')
    plt.ylabel(r'Memory Capacity')
    error_range = [0.1]*len(df_parity['node'])
    # plt.errorbar(df_parity['node'], df_parity['SUM'], c='red', label='parity', fmt='o:', ecolor='hotpink',
    # elinew1idth=3,
    #              ms=5, mfc='wheat', mec='salmon',yerr=error_range,
    #              capsize=3)
    plt.plot(df_parity['node'], df_parity['SUM'], c='red', label='Parity')
    plt.scatter(df_parity['node'], df_parity['SUM'], c='red')
    plt.legend()
    # plt.show()
    # sys.exit(-1)

    # #################################################################################################
    # making multi data excel files at different time to a same one (Comparison)
    # #################################################################################################
    plt.close('all')
    index_multi_data = True
    if index_multi_data:
        os.chdir('D:\\Python_projects\\Physical_RC_data\\best_time')
        time_list = [2e-9, 3e-9, 4e-9, 6e-9, 7e-9, 10e-9, 20e-9]
        task = 'Delay'
        final_value_list, error_list = [], []
        # for node 16
        for time_value in time_list:
            file_path = f'{task}_best_time_data_{time_value}_2.xlsx'
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                # print(df)
                max_df = np.sum(df['max']) + 1
                min_df = np.sum(df['min']) + 1
                error_df = (max_df - min_df)/2/((max_df + min_df)/2)
                final_value_list.append((max_df + min_df)/2)
                error_list.append(error_df)
            else:
                print('no file')

        # for other nodes
        task_list = ['Delay', 'Parity']
        number = np.linspace(0, 9, 10, dtype=int)
        # print(number)
        node_list = [10, 20, 30, 40, 50]

        for task in task_list:
            for node in node_list:
                capacity_list = np.zeros((4, 10))
                for time_value in time_list:
                    for number_value in number:
                        file_path = f'Finding_best_time_node_{node}_{number_value}_{time_value}_{task}_2021-11-08.xlsx'
                        another_file_path = f'Finding_best_time_node_{node}_{number_value}_{time_value}' \
                                            f'_{task}_2021-11-09.xlsx'
                        if os.path.exists(file_path):
                            file_final_path = file_path
                        elif os.path.exists(another_file_path):
                            file_final_path = another_file_path
                        else:
                            print('no data')

                        df = pd.read_excel(file_final_path)
                        capacity_list[:, int(number_value)] = df['covariance'].values
                    df = pd.DataFrame(capacity_list)
                    df.to_excel(f'Best_time_node_{node}_{task}_{time_value}.xlsx')

        plt.figure('Best time')
        plt.title(r'Best time for Delay task')
        plt.xlabel(r'Evolution Time')
        plt.ylabel(r'$MC_{STM}$')
        plt.ylim(0, 3)
        # plt.errorbar(time_list, final_value_list, yerr=error_list, fmt='o:', ecolor='red', mfc='blue', capsize=4,
        #              elinewidth=2, label='size 16')
        plt.scatter(time_list, final_value_list, label=f'size 16')
        plt.plot(time_list, final_value_list, linestyle='--')

        plt.figure('Best time')
        plt.title(r'Best time for Parity task')
        plt.xlabel(r'Evolution Time')
        plt.ylabel(r'Memory Capacity')
        plt.ylim(0, 3)
        task_list = ['Parity']
        # final_value_list = [1.48665877, 1.527006603, 1.590315218, 1.5370195, 1.489069385, 1.170078181, 1.117016993,
        #                     1.076723465]
        # plt.scatter(time_list, final_value_list, label=f' size 16')
        # plt.plot(time_list, final_value_list, linestyle='--')
        for task in task_list:
            for node in node_list:
                final_value_list, error_list = [], []
                for time_value in time_list:
                    file_path = f'Best_time_node_{node}_{task}_{time_value}.xlsx'
                    if os.path.exists(file_path):
                        df = pd.read_excel(file_path)
                        max_df = np.max(df.iloc[0, 1:]) + np.max(df.iloc[1, 1:]) + np.max(df.iloc[2, 1:]) + np.max(
                            df.iloc[3, 1:]) + 1
                        min_df = np.min(df.iloc[0, 1:]) + np.min(df.iloc[1, 1:]) + np.min(df.iloc[2, 1:]) + np.min(
                            df.iloc[3, 1:]) + 1
                        error_df = (max_df - min_df) / 2
                        final_value_list.append((max_df + min_df) / 2)
                        error_list.append(error_df)
                    else:
                        print('no file')
                # plt.errorbar(time_list, final_value_list, yerr=error_list, fmt='o:',
                #              capsize=2,
                #              elinewidth=1, label=f' size {node}')
                plt.scatter(time_list, final_value_list, label=f'size {node}')
                plt.plot(time_list, final_value_list, linestyle='--')

        # plt.figure('Best time')
        # plt.title(r'Best time for Delay task')
        # plt.xlabel(r'Evolution Time')
        # plt.ylabel(r'Memory Capacity')
        # plt.ylim(0, 3)

        # ax = plt.gca()
        # ax.set_yticks(time_list)
        plt.legend()
        plt.show()
        sys.exit(-1)

        # #################################################################################################
        # LOGICAL CALCULATION
        # #################################################################################################
        plt.close('all')
        os.chdir('D:\\Python_projects\\Physical_RC_data\\logical')
        df_xor = pd.read_excel('XOR2_from_0_100_number_0.xlsx')
        df_or = pd.read_excel('OR2_from_0_100_number_0.xlsx')
        df_and = pd.read_excel('AND2_from_0_100_number_0.xlsx')

        # plt.figure()
        # plt.subplot(3, 1, 1)
        plt.figure('OR_2 task', figsize=(4.8, 6.4))
        plt.title('OR_2 task')
        plt.scatter(df_or['le'], df_or['average'], c='green', label='OR')
        plt.xlabel(r'Time')
        plt.ylabel('Accuracy')
        plt.plot(x_zero_line, zero_line, c='black', label='zero line')
        x_major_locator = plt.MultipleLocator(2)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-11, 11)
        plt.ylim(0, 1)
        plt.legend()

        # plt.subplot(3, 1, 2)
        plt.figure('XOR_2 task', figsize=(4.8, 6.4))
        plt.title('XOR_2 task')
        plt.scatter(df_xor['le'], df_xor['average'], c='orange', label='XOR')
        plt.xlabel(r'Time')
        plt.ylabel('Accuracy')
        plt.plot(x_zero_line, zero_line, c='black', label='zero line')
        x_major_locator = plt.MultipleLocator(2)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-11, 11)
        plt.ylim(0, 1)
        plt.legend()

        # plt.subplot(3, 1, 3)
        plt.figure('AND_2 task', figsize=(4.8, 6.4))
        plt.scatter(df_and['le'], df_and['average'], c='blue', label='AND')

        plt.xlabel(r'Time')
        plt.ylabel('Accuracy')
        plt.plot(x_zero_line, zero_line, c='black', label='zero line')
        x_major_locator = plt.MultipleLocator(2)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-11, 11)
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

        # #################################################################################################
        # Multi MTJ figures
        # #################################################################################################
        # os.chdir('D:\\Python_projects\\Physical_RC_data\\multi')
        # df_oscillator_delay = pd.read_excel('Oscillator_node_20_Delay_0.xlsx')
        # df_oscillator_parity = pd.read_excel('Oscillator_node_20_Parity_0.xlsx')
        # df_switch_delay = pd.read_excel('Switch_node_20_Delay_0.xlsx')
        # df_switch_parity = pd.read_excel('Switch_node_20_Parity_0.xlsx')
        # df_multi_delay = pd.read_excel('Multi_node_20_Delay_0.xlsx')
        # df_multi_parity = pd.read_excel('Multi_node_20_Parity_0.xlsx')
        #
        # plt.close('all')
        # plt.figure('delay task')
        # plt.title('Delay task for multi-MTJ-system')
        # # plt.plot(df_switch_delay['number'], df_switch_delay['final'], label='switch', c='green', linestyle='--')
        # # plt.scatter(df_switch_delay['number'], df_switch_delay['final'], c='green')
        # error_y = (df_switch_delay['max'].values - df_switch_delay['min'].values)/2
        # plt.errorbar(df_switch_delay['number'], df_switch_delay['final'], yerr=error_y, fmt='o:', capsize=2,
        #              elinewidth=1, label='switch', ecolor='green', c='green')
        # plt.text(4, 0.6, s=f'Memory capacity={np.round(np.sum(df_switch_delay["final"]), 2)} (+-'
        #                    f'{np.round(np.sum(error_y), 2)})', c='green')
        # # plt.plot(df_oscillator_delay['number'], df_oscillator_delay['final'], c='orange', linestyle='--',
        # #          label='oscillator')
        # # plt.scatter(df_oscillator_delay['number'], df_oscillator_delay['final'], c='orange')
        # error_y = (df_oscillator_delay['max'].values - df_oscillator_delay['min'].values) / 2
        # plt.errorbar(df_oscillator_delay['number'], df_oscillator_delay['final'], yerr=error_y, fmt='o:', capsize=2,
        #              elinewidth=1, label='oscillator', ecolor='orange', c='orange')
        # plt.text(4, 0.65, s=f'Memory capacity={np.round(np.sum(df_oscillator_delay["final"]), 2)} (+-'
        #                     f'{np.round(np.sum(error_y), 2)})', c='orange')
        #
        # # plt.plot(df_multi_delay['number'], df_multi_delay['final'], c='blue', linestyle='--', label='mix')
        # # plt.scatter(df_multi_delay['number'], df_multi_delay['final'], c='blue')
        # error_y = (df_multi_delay['max'].values - df_multi_delay['min'].values) / 2
        # plt.errorbar(df_multi_delay['number'], df_multi_delay['final'], yerr=error_y, fmt='o:', capsize=2,
        #              elinewidth=1, label='mix', ecolor='blue', c='blue')
        # plt.text(4, 0.7, s=f'Memory capacity={np.round(np.sum(df_multi_delay["final"]), 2)} (+-'
        #                    f'{np.round(np.sum(error_y), 2)})', c='blue')
        #
        # plt.xlabel(r'Delay Time')
        # plt.ylabel(r'Capacity')
        # plt.legend()
        #
        # plt.figure('parity task')
        # plt.title('Parity task for multi-MTJ-system')
        # # plt.plot(df_switch_parity['number'], df_switch_parity['final'], label='switch', c='green', linestyle='--')
        # # plt.scatter(df_switch_parity['number'], df_switch_parity['final'], c='green')
        # error_y = (df_switch_parity['max'].values - df_switch_parity['min'].values) / 2
        # plt.errorbar(df_switch_parity['number'], df_switch_parity['final'], yerr=error_y, fmt='o:', capsize=2,
        #              elinewidth=1, label='switch', ecolor='green', c='green')
        # plt.text(4, 0.6, s=f'Memory capacity={np.round(np.sum(df_switch_parity["final"]), 2)} (+-'
        #                    f'{np.round(np.sum(error_y), 2)})', c='green')
        #
        # # plt.plot(df_oscillator_parity['number'], df_oscillator_parity['final'], c='orange', linestyle='--',
        # #          label='oscillator')
        # # plt.scatter(df_oscillator_parity['number'], df_oscillator_parity['final'], c='orange')
        # error_y = (df_oscillator_parity['max'].values - df_oscillator_parity['min'].values) / 2
        # plt.errorbar(df_oscillator_parity['number'], df_oscillator_parity['final'], yerr=error_y, fmt='o:', capsize=2,
        #              elinewidth=1, label='oscillator', ecolor='orange', c='orange')
        # plt.text(4, 0.65, s=f'Memory capacity={np.round(np.sum(df_oscillator_parity["final"]), 2)} (+-'
        #                     f'{np.round(np.sum(error_y), 2)})', c='orange')
        #
        # # plt.plot(df_multi_parity['number'], df_multi_parity['final'], c='blue', linestyle='--', label='mix')
        # # plt.scatter(df_multi_parity['number'], df_multi_parity['final'], c='blue')
        # error_y = (df_multi_parity['max'].values - df_multi_parity['min'].values) / 2
        # plt.errorbar(df_multi_parity['number'], df_multi_parity['final'], yerr=error_y, fmt='o:', capsize=2,
        #              elinewidth=1, label='mix', ecolor='blue', c='blue')
        # plt.text(4, 0.7, s=f'Memory capacity={np.round(np.sum(df_multi_parity["final"]), 2)} (+-'
        #                    f'{np.round(np.sum(error_y), 2)})', c='blue')
        #
        # plt.xlabel(r'Delay Time')
        # plt.ylabel(r'Capacity')
        # # plt.text(4, 0.6, s=f'capacity={np.round(np.sum(df_switch_parity["final"]), 2)}', c='green')
        # # plt.text(4, 0.65, s=f'capacity={np.round(np.sum(df_oscillator_parity["final"]), 2)}', c='orange')
        # # plt.text(4, 0.7, s=f'capacity={np.round(np.sum(df_multi_parity["final"]), 2)}', c='blue')
        # plt.legend()
        # plt.show()

        # #################################################################################################
        # Multi MTJ put data into one file
        # #################################################################################################
        # os.chdir('D:\\Python_projects\\Physical_RC_data\\multi\\switch_nodes')
        # node_list = [2, 5, 10, 16, 30, 50]
        # superposition_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # task_list = ['Delay', 'Parity']
        #
        # for node in node_list:
        #     for task in task_list:
        #         covariance_list = np.zeros((10, 10))
        #         for number_value in range(10):
        #             file_path = f'Oscillator_node_{node}_{task}_{number_value}.xlsx'
        #             if os.path.exists(file_path):
        #                 df = pd.read_excel(file_path)
        #                 covariance_list[:, number_value] = df['covariance'].values
        #
        #         df = pd.DataFrame(covariance_list)
        #         df.to_excel(f'Switch_node_{node}_{task}.xlsx')

        # #################################################################################################
        # Multi MTJ draw figures
        # #################################################################################################
        # os.chdir('D:\\Python_projects\\Physical_RC_data\\multi\\switch_nodes')
        # file_path = 'Switch_nodes_diff_Delay.xlsx'
        # if os.path.exists(file_path):
        #     df = pd.read_excel(file_path)
        #     plt.figure('MC nodes diff')
        #     plt.title(r'different nodes for Multi-MTJ in Delay task')
        #     plt.plot(df['nodes'], df['MC'], c='red', label='Switch')
        #     plt.scatter(df['nodes'], df['MC'], c='red')
        #     plt.xlabel('nodes')
        #     plt.ylabel('Memory capacity')
        #     plt.legend()
            # plt.show()


