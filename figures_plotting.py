import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # fig
    plt.figure()
    consuming_time_list = [1e-9]*20
    consuming_time_list = [1, 2, 3, 4]
    plt.xlabel(r'DC current(Oe)', size=16)
    plt.ylabel(r'$M_{z}$ amplitude (a.u.)', size=16)
    save_data_path='input_configuration_data/error_bar_random_initial'
    for i in range(len(consuming_time_list)):
        file_name = f'{save_data_path}/input_configuraiton_1.0e-9_{i+21}.xlsx'
        raw_data = pd.read_excel(file_name)
        data = raw_data.drop(raw_data[raw_data['invalid']==False].index)
        plt.plot(data['dc_current'], data['mz_amplitude_max'], label=f'curve {i}')
    
    # index line
    x = [90]*20
    y = np.linspace(-1, 1, 20)
    plt.plot(x, y, '-', linewidth=5, alpha=0.5)

    x = [300]*20
    y = np.linspace(-1, 1, 20)
    plt.plot(x, y, '-', linewidth=5, alpha=0.5)

    plt.ylim(0, 0.55)
    plt.legend()
    plt.show()

    # ####################################################
    # different ration of inputs
    # ####################################################

    posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1)
    nodes = [16, 20, 30, 40]
    superposition_list = [1, 2, 3, 4]
    save_data_path = f'data_stochastic_'
    # plt.figure()
    # plt.xlabel(r'$T_{delay}$', size=16)
    # plt.ylabel(r'$Cor^{2}$', size=16)
    for node in nodes:
        plt.figure()
        plt.xlabel(r'$T_{delay}$', size=16)
        plt.ylabel(r'$Cor^{2}$', size=16)
        plt.title(r'Short Term Memory Task', size=12)
        for posibiliy in posibility_list:
            data_name = f'data_positibility{posibiliy}_node_{node}_3e-9.csv'
            data = pd.read_csv(f'{save_data_path}/{data_name}')
            plt.plot(data['superposition'], data['cor_2_delay'], label=f'node:{node} posibility: {posibiliy}')
            plt.scatter(data['superposition'], data['cor_2_delay'])
    
        plt.legend()
        plt.show()

    for node in nodes:
        plt.figure()
        plt.xlabel(r'$T_{delay}$', size=16)
        plt.ylabel(r'$Cor^{2}$', size=16)
        plt.title(r'Parity Check Task', size=12)
        for posibiliy in posibility_list:
            data_name = f'data_positibility{posibiliy}_node_{node}_3e-9.csv'
            data = pd.read_csv(f'{save_data_path}/{data_name}')
            plt.plot(data['superposition'], data['cor_2_pc'], label=f'node:{node} posibility: {posibiliy}')
            plt.scatter(data['superposition'], data['cor_2_pc'])
    
        plt.legend()
        plt.show()
