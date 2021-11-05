"""
This code is used to draw all of figures and data in Physical reservoir computing project
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        df_delay = pd.ExcelFile('E:\\磁\\data\\Delay.xlsx')
        df_delay_0 = pd.read_excel(df_delay, 'delay0')
        df_delay_1 = pd.read_excel(df_delay, 'delay1')
        df_delay_2 = pd.read_excel(df_delay, 'delay2')

        df_parity = pd.ExcelFile('E:\\磁\\data\\Parity.xlsx')
        df_parity_0 = pd.read_excel(df_parity, 'parity0')
        df_parity_1 = pd.read_excel(df_parity, 'parity1')
        df_parity_2 = pd.read_excel(df_parity, 'parity2')

        # # lin shi
        # df_lyapunov = pd.read_excel('E:\\磁\\data\\lyapunov_reservoir.xlsx')
        # df_parity_0 = pd.merge(df_lyapunov, df_parity_0)
        # df_parity_1 = pd.merge(df_lyapunov, df_parity_1)
        #
        # zero_line = np.linspace(-0.1, 1.1, 20)
        # x_zero_line = [0] * len(zero_line)
        # plt.figure('Parity task - lyapunov')
        # plt.plot(x_zero_line, zero_line, c='black', label='zero line')
        # # plt.scatter(df_parity_0['le'], df_parity_0['average'], label='parity0')
        # # plt.scatter(df_parity_1['le'], df_parity_1['average'], label='parity1')
        # plt.scatter(df_parity_2['lyapunov'], df_parity_2['average'], label='parity2', c='green')
        # plt.xlabel('Lyapunov exponent')
        # plt.ylabel('Capacity')
        # plt.legend()
        # x_major_locator = plt.MultipleLocator(10)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_major_locator)
        # plt.xlim(-11, 11)
        # plt.show()

        # df_lyapunov = pd.read_excel('E:\\磁\\data\\lyapunov_mag.xlsx')
        df_lyapunov = pd.read_excel('E:\\磁\\data\\lyapunov_reservoir.xlsx')
        df_parity_0 = pd.merge(df_lyapunov, df_parity_0)
        df_parity_1 = pd.merge(df_lyapunov, df_parity_1)
        df_parity_2 = pd.merge(df_lyapunov, df_parity_2)
        df_delay_0 = pd.merge(df_lyapunov, df_delay_0)
        df_delay_1 = pd.merge(df_lyapunov, df_delay_1)
        df_delay_2 = pd.merge(df_lyapunov, df_delay_2)
        df = df_parity_2[['ac', 'le', 'average']].drop_duplicates()
        df = df.sort_values(by="ac", ignore_index=True)
        df.to_excel('E:\\磁\\data\\result.xlsx')

        # zero line drawing
        zero_line = np.linspace(-0.1, 1.1, 20)
        x_zero_line = [0] * len(zero_line)

        # normalization and calculation of total capacity
        capacity_delay = np.add(df_delay_0['average'].values, df_delay_1['average'].values)
        capacity_delay = np.add(capacity_delay, df_delay_2['average'].values)
        capacity_delay = capacity_delay / max(capacity_delay)

        capacity_parity = np.add(df_parity_0['average'].values, df_parity_1['average'].values)
        # capacity_parity = np.add(capacity_parity, df_parity_2['average'].values)
        capacity_parity = capacity_parity / max(capacity_parity)

        plt.figure('whole capacity')
        plt.plot(x_zero_line, zero_line, c='black', label='zero line')
        plt.scatter(df_delay_0['le'], capacity_delay, label='delay')
        plt.scatter(df_parity_0['le'], capacity_parity, label='parity')
        plt.xlabel('Lyapunov exponent')
        plt.ylabel('Capacity')
        plt.legend()
        x_major_locator = plt.MultipleLocator(10)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-11, 11)
        # plt.ylim(0, 1)

        plt.figure('delay task - lyapunov')
        plt.plot(x_zero_line, zero_line, c='black', label='zero line')
        plt.scatter(df_delay_0['le'], df_delay_0['average'], label='delay0')
        plt.scatter(df_delay_1['le'], df_delay_1['average'], label='delay1')
        plt.scatter(df_delay_2['le'], df_delay_2['average'], label='delay2')
        plt.xlabel('Lyapunov exponent')
        plt.ylabel('Capacity')
        plt.legend()
        x_major_locator = plt.MultipleLocator(10)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlim(-11, 11)
        # plt.ylim(0, 1)

        plt.figure('Parity task - lyapunov')
        plt.plot(x_zero_line, zero_line, c='black', label='zero line')
        plt.scatter(df_parity_0['le'], df_parity_0['average'], label='parity0')
        plt.scatter(df_parity_1['le'], df_parity_1['average'], label='parity1')
        plt.scatter(df_parity_2['le'], df_parity_2['average'], label='parity2')
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
    time_list = [6e-9, 7e-9, 8e-9, 1e-8, 5e-9, 9e-9]
    superposition_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    task_list = ['Delay', 'Parity']
    for task in task_list:
        for time_value in time_list:
            file_name = f'Finding_best_time_{time_value}_{task}_0.xlsx'
            if os.path.exists(file_name):
                data_value = np.zeros((5, len(superposition_list)))
                for superposition_number in superposition_list:
                    file_path = f'Finding_best_time_{time_value}_{task}_{superposition_number}.xlsx'
                    data_value[:, superposition_number] = pd.read_excel(file_path)['covariance']
                    # print(file_path)
            new_data = pd.DataFrame(data_value)
            new_data.to_excel(f'{task}_best_time_data_{time_value}.xlsx')

    # #################################################################################################
    # drawing figures of best reservoir size
    # #################################################################################################
    df_parity = pd.read_excel('E:\\磁\\data\\Best_size_Parity.xlsx')
    df_delay = pd.read_excel('E:\\磁\\data\\Best_size_delay.xlsx')

    plt.figure('Best reservoir size for delay task')
    plt.title('Best reservoir size')
    plt.xlabel(r'reservoir size')
    plt.ylabel(r'Memory Capacity')
    plt.plot(df_delay['node'], df_delay['sum'], c='blue', label='Delay')
    plt.scatter(df_delay['node'], df_delay['sum'], c='blue')
    plt.ylim(0, 3)
    plt.legend()

    plt.figure('Best reservoir size for parity task')
    plt.title('Best reservoir size')
    plt.xlabel(r'reservoir size')
    plt.ylabel(r'Memory Capacity')
    error_range = [0.1]*len(df_parity['node'])
    # plt.errorbar(df_parity['node'], df_parity['SUM'], c='red', label='parity', fmt='o:', ecolor='hotpink', elinewidth=3,
    #              ms=5, mfc='wheat', mec='salmon',yerr=error_range,
    #              capsize=3)
    plt.plot(df_parity['node'], df_parity['SUM'], c='red', label='Parity')
    plt.scatter(df_parity['node'], df_parity['SUM'], c='red')
    plt.legend()
    plt.show()

    #
    df = pd.read_excel('E:\\磁\\data\\Best_time_parity.xlsx')
