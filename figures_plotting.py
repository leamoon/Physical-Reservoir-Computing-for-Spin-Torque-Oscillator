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
    x = [91]*len()
    plt.legend()
    plt.show()
