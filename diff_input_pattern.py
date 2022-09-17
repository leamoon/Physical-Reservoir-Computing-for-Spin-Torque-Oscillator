import mtj_module
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin
from rich.progress import track
from multiprocessing import Pool
import itertools

"""
    This code is used to detect the influence of different input patterns. (The ratio of 1 & 0)
    Author: Xuezhao WU
    Date: 2022.Sep.16
"""

# random seed, help for production
np.random.seed(110)

def input_current_task_train(node=20, posibility_0=0.5, superposition=2, time_comsuming=3e-9):
    initial_m = np.random.random(3)
    device = mtj_module.Mtj(initial_m[0], initial_m[1], initial_m[2])
    device.real_time_train(number_wave=1000, nodes_stm=node, file_path=f'weight_evolution_{time_comsuming}/weight_posibility_{posibility_0}', superposition=superposition, time_consume_all=time_comsuming,
                           task='Delay', visual_process=False, recover_weight=False, posibility_0=posibility_0)
    device.real_time_train(number_wave=1000, nodes_stm=node, file_path=f'weight_evolution_{time_comsuming}/weight_posibility_{posibility_0}', superposition=superposition, time_consume_all=time_comsuming,
                           task='Parity', visual_process=False, recover_weight=False, posibility_0=posibility_0)
    


def input_current_task_test(node_list, posibility_0, superposition_list, time_comsuming=3e-9):
    initial_m = np.random.random(3)
    device = mtj_module.Mtj(initial_m[0], initial_m[1], initial_m[2])
    for node in node_list:
        cor_list = np.zeros((2, len(superposition_list)))
        for i in range(len(superposition_list)):
            superposition = superposition_list[i]
            capacity_delay = device.real_time_test(test_number=100, nodes_stm=node, file_path=f'weight_evolution_{time_comsuming}/weight_posibility_{posibility_0}', superposition=superposition, visual_index=False, 
                                                time_consume_all=time_comsuming, task='Delay', posibility_0=posibility_0)
            capacity_parity = device.real_time_test(test_number=100, nodes_stm=node, file_path=f'weight_evolution_{time_comsuming}/weight_posibility_{posibility_0}', superposition=superposition, visual_index=False, 
                                                time_consume_all=time_comsuming, task='Parity', posibility_0=posibility_0)
            cor_list[0, i] = capacity_delay
            cor_list[1, i] = capacity_parity
        
        data = pd.DataFrame({'cor_2_delay': cor_list[0, :], 'cor_2_pc': cor_list[1, :], 'superposition': superposition_list})
        file_save_path = 'data_stochastic_'
        if not os.path.exists(file_save_path):
            os.makedirs(file_save_path)
        file_name = f'data_positibility{posibility_0}_node_{node}_{time_comsuming}.csv'
        data.to_csv(f'{file_save_path}/{file_name}')

            

if __name__ == '__main__':

    # test for different pair
    posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1)
    nodes = [16, 20, 30, 40]
    nodes = [2, 5, 10, 16, 20, 30, 40, 50, 70, 90, 100]
    superposition_list = [1, 2, 3, 4]
    superposition_list = np.linspace(1, 10, 10, dtype=int)
    time_list = [2e-9, 4e-9, 6e-9, 7e-9, 10e-9, 20e-9]
    time_list = [7e-9]
    # print(superposition_list)
    # sys.exit()
    # print(posibility_list)
    # for posibility in posibility_list:
    #     input_current_task_test(node_list=nodes, posibility_0=posibility, superposition_list=superposition_list)
    for evolution_time in track(time_list):
        for node in nodes:
            for superposition in superposition_list:
                with Pool() as pool:
                    pool.starmap(input_current_task_train, 
                                zip(itertools.repeat(node), posibility_list, itertools.repeat(superposition), itertools.repeat(evolution_time)))

    # test
    for evolution_time in time_list:
        for i in posibility_list:
            input_current_task_train(posibility_0=i, time_comsuming=evolution_time)
    
    mtj_module.email_alert('7e-9 task is finished !')