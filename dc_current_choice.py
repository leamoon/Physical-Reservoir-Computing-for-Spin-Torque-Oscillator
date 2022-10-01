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
    This code has some functions to detect the STM and PC tasks performace under different dc current pairs.
    Author: Xuezhao WU
    Date: 2022.Sep.16
"""

# random seed, help for production
# np.random.seed(110)

def input_current_task_test(positive_dc_current, negative_dc_current):
    initial_m = np.random.random(3)
    device = mtj_module.Mtj(initial_m[0], initial_m[1], initial_m[2])
    device.real_time_train(number_wave=1000, nodes_stm=20, file_path=f'weight_dc_{positive_dc_current}_{negative_dc_current}', superposition=2, time_consume_all=3e-9,
                           task='Delay', positive_dc_current=positive_dc_current, negative_dc_current=negative_dc_current, visual_process=False, recover_weight=False)
    device.real_time_train(number_wave=1000, nodes_stm=20, file_path=f'weight_dc_{positive_dc_current}_{negative_dc_current}', superposition=2, time_consume_all=3e-9,
                           task='Parity', positive_dc_current=positive_dc_current, negative_dc_current=negative_dc_current, visual_process=False, recover_weight=False)
    # device.real_time_test(test_number=50, nodes_stm=20, file_path=f'weight_dc_{positive_dc_current}_{negative_dc_current}', superposition=2, visual_index=True, 
    #                       time_consume_all=3e-9, task='Delay', positive_dc_current=positive_dc_current, negative_dc_current=negative_dc_current)

def collect_results(positive_dc_current_list, negative_dc_current_list):
    initial_m = np.random.random(3)
    device = mtj_module.Mtj(initial_m[0], initial_m[1], initial_m[2])
    cor_list = np.zeros((len(positive_dc_current_list), 1))
    for i in range(len(positive_dc_current_list)):
        cor_2 = device.real_time_test(test_number=50, nodes_stm=20, file_path=f'weight_dc_{positive_dc_current_list[i]}_{negative_dc_current_list[i]}', superposition=2, visual_index=False, 
                          time_consume_all=3e-9, task='Delay', positive_dc_current=positive_dc_current_list[i], negative_dc_current=negative_dc_current_list[i])
        
        cor_list[i, 0] = cor_2
        file_save_path = f'diff_pair_current_50_positive_node20_superposition2_Delay_3e-9.csv'
        df = pd.DataFrame({'positive': positive_dc_current_list, 'negative': negative_dc_current_list, 'cor_2': cor_list[:, 0]})
        df.to_csv(file_save_path)

if __name__ == '__main__':
    # dc current (100, 140), (120, 140), (200, 400), (200, 300), (140, 400), (100, 200)
    # input_current_task_test(positive_dc_current=400, negative_dc_current=500)


    # positive_dc_current_list = [100, 120, 200, 200, 140, 100]
    # negative_dc_current_list = [140, 140, 400, 300, 400, 200]
    # collect_results(positive_dc_current_list=positive_dc_current_list, negative_dc_current_list=negative_dc_current_list)

    
    negative_dc_current_list = np.linspace(120, 1000, 50, dtype=int)
    positive_dc_current_list = [50]*len(negative_dc_current_list)

    # print(negative_dc_current_list)

    # test code
    # for i in range(len(positive_dc_current_list)):
    #     input_current_task_test(positive_dc_current=positive_dc_current_list[i], negative_dc_current=negative_dc_current_list[i])

    with Pool() as pool:
        pool.starmap(input_current_task_test, 
                     zip(positive_dc_current_list, negative_dc_current_list))
    
    collect_results(positive_dc_current_list=positive_dc_current_list, negative_dc_current_list=negative_dc_current_list)