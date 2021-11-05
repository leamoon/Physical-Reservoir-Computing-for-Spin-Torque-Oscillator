import os
import sys
from edge_esn_oscillator import *
import pandas as pd
import numpy as np
from rich.progress import track


if __name__ == '__main__':
    # change working path
    work_path = 'D:\\Python_projects\\Physical_RC_data'
    if os.path.exists(work_path):
        os.chdir(work_path)
    print(f'Path: {work_path}')

    # ##################################################################################################
    # collecting real time task result
    # ##################################################################################################
    # ac_list = np.round(np.linspace(25, 41.9, 170), 1)
    # print(ac_list)
    # covariance_list = []
    # for ac_value in ac_list:
    #     mtj = Mtj()
    #     mtj.real_time_train(number_wave=10, nodes_stm=16, visual_process=True, save_index=False,
    #                         alert_index=False, superposition=2, time_consume_all=1e-9,
    #                         file_path='weight_matrix_interpolation',
    #                         ac_amplitude=ac_value, task='Parity', recover_weight=True)
    #
    #     capacity = mtj.real_time_test(test_number=100, nodes_stm=16, superposition=2,
    #                                   visual_index=False, file_path='weight_matrix_interpolation',
    #                                   ac_amplitude=ac_value, time_consume_all=6e-9, task='Parity')
    #     print(capacity)
    #     covariance_list.append(capacity)
    # df = pd.DataFrame({'ac': ac_list, 'capacity': covariance_list})
    # df.to_excel('parity2_25_41.9.xlsx')

    # ##################################################################################################
    # #### find best evolution time
    get_best_time(task='Delay', re_train=True)
    # get_best_time(task='Parity', re_train=True)
    # ##################################################################################################
    # #### find best reservoirs info: size
    # get_best_reservoir_info(task='Delay', max_time=6)
    # get_best_reservoir_info(task='Parity', max_time=6)

    # ##########################################################################################
    # finding critical line
    # ##########################################################################################
    # variable: ac amplitude
    # try:
    #     df = pd.read_excel('lyapunov_mag.xlsx')
    #     ac_exist_term = np.round(df['ac'].values, 1)
    #     ac_target_list = np.round(np.linspace(0, 100, 1001), 1)
    #
    #     ac_amplitude_list = [i for i in ac_target_list if i not in ac_exist_term]
    #
    #     largest_lyapunov_exponent = []
    #     frequency_ac_term_list = 32e9
    #
    #     for ac_stt in track(range(len(ac_amplitude_list))):
    #         mle = edge_esn_oscillator.chaos_mine(ac_current1=ac_amplitude_list[ac_stt], f_ac=frequency_ac_term_list,
    #                                              size=16,
    #                                              time_consume_single=1e-8,
    #                                              input_mode='periodic', save_as_excel=True)
    #         largest_lyapunov_exponent.append(mle)
    #
    # except Exception as error_message:
    #     sys.exit(error_message)
