import itertools
import os
import sys
import time
from multiprocessing import Pool

from edge_esn_oscillator import *
import pandas as pd
import numpy as np
from rich.progress import track


# def train(ac_value):
#     mtj = Mtj()
#     mtj.real_time_train(number_wave=500, nodes_stm=16, visual_process=False, save_index=True,
#                         alert_index=False, superposition=1, time_consume_all=4e-9,
#                         file_path='weight_matrix_interpolation',
#                         ac_amplitude=ac_value, task='XOR', recover_weight=False)
#     mtj.real_time_train(number_wave=500, nodes_stm=16, visual_process=False, save_index=True,
#                         alert_index=False, superposition=1, time_consume_all=4e-9,
#                         file_path='weight_matrix_interpolation',
#                         ac_amplitude=ac_value, task='AND', recover_weight=False)
#     mtj.real_time_train(number_wave=500, nodes_stm=16, visual_process=False, save_index=True,
#                         alert_index=False, superposition=1, time_consume_all=4e-9,
#                         file_path='weight_matrix_interpolation',
#                         ac_amplitude=ac_value, task='OR', recover_weight=False)
#     print(f'ac_{ac_value} finished')
#
#
def test_edge(ac_value, task):
    # a function used to test the edge of chaos with different superposition values
    capacity_list = []
    for i in range(10):
        mtj = Mtj()
        capacity = mtj.real_time_test(test_number=150, nodes_stm=16, superposition=3,
                                      visual_index=False,
                                      file_path='weight_matrix_interpolation',
                                      ac_amplitude=ac_value, time_consume_all=4e-9, task=task)
        capacity_list.append(capacity)

    # save as excel files
    df = pd.DataFrame({'capacity': capacity_list})
    df.to_excel(f'delay_time_3_ac{ac_value}_{task}_node16_oscillator.xlsx')

    capacity_list = []
    for i in range(10):
        mtj = Mtj()
        capacity = mtj.real_time_test(test_number=150, nodes_stm=16, superposition=4,
                                      visual_index=False,
                                      file_path='weight_matrix_interpolation',
                                      ac_amplitude=ac_value, time_consume_all=4e-9, task=task)
        capacity_list.append(capacity)

    # save as excel files
    df = pd.DataFrame({'capacity': capacity_list})
    df.to_excel(f'delay_time_4_ac{ac_value}_{task}_node16_oscillator.xlsx')


if __name__ == '__main__':
    # change working path
    work_path = 'D:\\Python_projects\\Physical_RC_data'
    if os.path.exists(work_path):
        os.chdir(work_path)
    elif os.path.exists('/home/xwubx/get_weight'):
        os.chdir('/home/xwubx/get_weight')
    print(f'Path: {os.getcwd()}')
    # for i in range(10):
    #     mtj = Mtj()
    #     capacity = mtj.real_time_test(test_number=150, nodes_stm=16, visual_index=False, superposition=3,
    #                                   ac_amplitude=0.0, task='Delay', file_path='weight_matrix_interpolation',
    #                                   time_consume_all=4e-9)
    #     print(capacity)
    # lyapunov_numeric(ac_amplitude=0, dc_amplitude=260)

    # test for multiprocessing
    # superposition_list = np.linspace(0, 10, 11, dtype=int)
    # node_list = [2, 5, 10, 20, 30, 40, 50, 100]
    # t1 = time.time()
    # with Pool() as pool:
    #     for node in node_list:
    #         pool.starmap(train, zip(itertools.repeat(node), superposition_list))
    #
    # with Pool() as pool:
    #     pool.map(test, node_list)
    # t2 = time.time()
    # print('{:.2f}s'.format(t2 - t1))

    ac_list = np.round(np.linspace(0, 100, 1001), 1)
    super_position_list = [3, 4]
    task_list = ['Delay', 'Parity']
    print(ac_list)
    with Pool() as pool:
        for task in task_list:
            pool.starmap(test_edge, zip(ac_list, itertools.repeat(task)))

    email_alert(subject='Task finished')

    # sys.exit()
    # ##################################################################################################
    # calculate lyapunov exponent
    # ##################################################################################################
    # chaos_mine(initial_dif=1e-8, ac_current1=0.1, input_mode='periodic', time_consume_single=4e-9, size=16)
    # chaos_transition(ac_current1=0.0)

    # ##################################################################################################
    # collecting real time task result
    # ##################################################################################################
    # ac_list = np.round(np.linspace(0, 100, 1001), 1)
    # capacity_list, ac_value_list = [], []
    # print(ac_list)
    # for ac_value in ac_list:
    #     mtj = Mtj()
    #     mtj.transition_ability(length_signals=70, node_transition=16, visual_index=False, ac_amplitude=ac_value,
    #                            recover_weight=False, save_index=True, task='Square')
    #     capacity = mtj.test_of_transition(length_signals=10, node_transition=16, visual_index=False,
    #                                       ac_amplitude=ac_value, task='Square')
    #     capacity_list.append(capacity)
    #     ac_value_list.append(ac_value)
    #     df = pd.DataFrame({'ac': ac_value_list, 'capacity': capacity_list})
    #     df.to_excel('Square_transition_result.xlsx')
    #
    # email_alert(subject='training successfully!')
    # ##################################################################################################
    # collecting real time task result
    # ##################################################################################################
    # ac_list = np.round(np.linspace(0, 100, 1001), 1)
    # super_position_list = [3, 4]
    # task_list = ['Delay', 'Parity']
    # print(ac_list)
    # weight_path = os.getcwd()
    # for i in range(9):
    #     # change working path
    #     data_save_path = os.path.join(os.getcwd(), f'data_{i}')
    #     if not os.path.exists(data_save_path):
    #         os.mkdir(data_save_path)
    #     os.chdir(data_save_path)
    #     print('path changed successfully !')
    #     print(f'current path: {os.getcwd()}')
    #
    #     for task in task_list:
    #         for super_value in super_position_list:
    #             covariance_list, ac_value_list = [], []
    #             for ac_value in ac_list:
    #                 mtj = Mtj()
    #                 mtj.real_time_train(number_wave=500, nodes_stm=16, visual_process=False, save_index=True,
    #                                     alert_index=False, superposition=super_value, time_consume_all=4e-9,
    #                                     file_path=os.path.join(weight_path, 'weight_matrix_interpolation'),
    #                                     ac_amplitude=ac_value, task=task, recover_weight=False)
    #
    #                 capacity = mtj.real_time_test(test_number=100, nodes_stm=16, superposition=super_value,
    #                                               visual_index=True,
    #                                               file_path=os.path.join(weight_path, 'weight_matrix_interpolation'),
    #                                               ac_amplitude=ac_value, time_consume_all=4e-9, task=task)
    #                 print(capacity)
    #                 covariance_list.append(capacity)
    #                 ac_value_list.append(ac_value)
    #                 df = pd.DataFrame({'ac': ac_value_list, 'capacity': covariance_list})
    #                 df.to_excel(f'{task}{super_value}_number_{i}.xlsx')
    #
    #     os.chdir(weight_path)
    # ##################################################################################################
    # #### find best evolution time
    # get_best_time(task='Delay', re_train=False, node=30)
    # get_best_time(task='Parity', re_train=False, node=30)
    # ##################################################################################################
    # #### find best reservoirs info: size
    # get_best_reservoir_info(task='Delay', max_time=6)
    # get_best_reservoir_info(task='Parity', max_time=6)

    # ##########################################################################################
    # finding critical line
    # ##########################################################################################
    # variable: ac amplitude
    # try:
    #     largest_lyapunov_exponent, ac_value_list = [], []
    #     if os.path.exists('lyapunov_mag.xlsx'):
    #         df = pd.read_excel('lyapunov_mag.xlsx')
    #         ac_exist_term = np.round(df['ac'].values, 1)
    #         ac_value_list = ac_value_list + list(ac_exist_term)
    #         largest_lyapunov_exponent = largest_lyapunov_exponent + list(df['le'].values)
    #     else:
    #         ac_exist_term = []
    #
    #     ac_target_list = np.round(np.linspace(0, 100, 1001), 1)
    #     ac_amplitude_list = [i for i in ac_target_list if i not in ac_exist_term]
    #
    #     frequency_ac_term_list = 32e9
    #
    #     for ac_stt in track(range(len(ac_amplitude_list))):
    #         mle = chaos_mine(ac_current1=ac_amplitude_list[ac_stt], f_ac=frequency_ac_term_list,
    #                          size=16,
    #                          time_consume_single=4e-9,
    #                          input_mode='periodic', save_as_excel=True)
    #         largest_lyapunov_exponent.append(mle)
    #         ac_value_list.append(ac_amplitude_list[ac_stt])
    #         df = pd.DataFrame({'ac': ac_value_list, 'le': largest_lyapunov_exponent})
    #         df.to_excel('lyapunov_mag.xlsx')
    #
    # except Exception as error_message:
    #     sys.exit(error_message)

    # for transition task
    # try:
    #     largest_lyapunov_exponent, ac_value_list = [], []
    #     if os.path.exists('lyapunov_mag_transition.xlsx'):
    #         df = pd.read_excel('lyapunov_mag_transition.xlsx')
    #         ac_exist_term = np.round(df['ac'].values, 1)
    #         ac_value_list = ac_value_list + list(ac_exist_term)
    #         largest_lyapunov_exponent = largest_lyapunov_exponent + list(df['le'].values)
    #     else:
    #         ac_exist_term = []
    #
    #     ac_target_list = np.round(np.linspace(0, 100, 1001), 1)
    #     ac_amplitude_list = [i for i in ac_target_list if i not in ac_exist_term]
    #
    #     frequency_ac_term_list = 32e9
    #
    #     for ac_stt in track(range(len(ac_amplitude_list))):
    #         mle = chaos_transition(ac_current1=ac_amplitude_list[ac_stt], f_ac=frequency_ac_term_list,
    #                                size=16,
    #                                time_consume_single=4e-9,
    #                                save_as_excel=True)
    #         largest_lyapunov_exponent.append(mle)
    #         ac_value_list.append(ac_amplitude_list[ac_stt])
    #         df = pd.DataFrame({'ac': ac_value_list, 'le': largest_lyapunov_exponent})
    #         df.to_excel('lyapunov_mag_transition.xlsx')
    #
    # except Exception as error_message:
    #     sys.exit(error_message)
