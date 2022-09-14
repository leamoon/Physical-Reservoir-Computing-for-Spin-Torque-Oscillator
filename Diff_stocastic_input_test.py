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

# random seed, help for production
np.random.seed(110)

def input_current_task_train(node=20, posibility_0=0.5, superposition=2):
    initial_m = np.random.random(3)
    device = mtj_module.Mtj(initial_m[0], initial_m[1], initial_m[2])
    device.real_time_train(number_wave=1000, nodes_stm=node, file_path=f'weight_posibility_{posibility_0}', superposition=superposition, time_consume_all=3e-9,
                           task='Delay', visual_process=False, recover_weight=False, posibility_0=posibility_0)
    device.real_time_train(number_wave=1000, nodes_stm=node, file_path=f'weight_posibility_{posibility_0}', superposition=superposition, time_consume_all=3e-9,
                           task='Parity', visual_process=False, recover_weight=False, posibility_0=posibility_0)
    


def input_current_task_test(node_list, posibility_0, superposition_list):
    initial_m = np.random.random(3)
    device = mtj_module.Mtj(initial_m[0], initial_m[1], initial_m[2])
    for node in node_list:
        for superposition in superposition_list:
            capacity_delay = device.real_time_test(test_number=50, nodes_stm=node, file_path=f'weight_posibility_{posibility_0}', superposition=superposition, visual_index=True, 
                                                time_consume_all=3e-9, task='Delay', posibility_0=posibility_0)
            capacity_parity = device.real_time_test(test_number=50, nodes_stm=node, file_path=f'weight_posibility_{posibility_0}', superposition=superposition, visual_index=True, 
                                                time_consume_all=3e-9, task='Parity', posibility_0=posibility_0)

            

if __name__ == '__main__':
    # dc current (100, 140), (120, 140), (200, 400), (200, 300), (140, 400), (100, 200)
    # input_current_task_test(positive_dc_current=100, negative_dc_current=140)
    posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1)
    nodes = [16, 20, 30, 40]
    superposition_list = [1, 2, 3, 4]
    print(posibility_list)
    for node in nodes:
        for superposition in superposition_list:
            with Pool() as pool:
                pool.starmap(input_current_task_train, 
                            zip(itertools.repeat(node), posibility_list, itertools.repeat(superposition)))

    # test
    # for i in posibility_list:
    #     input_current_task_train(posibility_0=i)