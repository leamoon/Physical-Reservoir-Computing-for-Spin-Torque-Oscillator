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
from scipy.interpolate import interp1d

def esp_index_calculation(random_initial_states=50, length_input=1000, ratio=0.5, consuming_time=4e-9, ac_current=0.0):
    # initialize the mtj module and m
    s_in = np.random.choice([0, 1], size=length_input, p=[ratio, 1-ratio])
    save_path = f'esp_data/{ratio}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for random_index in range(random_initial_states):
        if os.path.exists(f'{save_path}/ratio_{ratio}_{consuming_time}_{ac_current}_random_{random_index}.csv'):
            print('random input files already exit')
            continue
        
        initial_m = np.random.random(3)
        device = mtj_module.Mtj(initial_m[0], initial_m[1], initial_m[2])
        reservoir_data = {'s_in': s_in, 'magnetization': initial_m}

        for i in track(range(len(s_in))):
            if s_in[i] == 0:
                _, _, mz_list, _, _ = device.time_evolution(dc_amplitude=100, time_consumed=consuming_time, ac_amplitude=ac_current)
            else:
                _, _, mz_list, _, _ = device.time_evolution(dc_amplitude=200, time_consumed=consuming_time, ac_amplitude=ac_current)

            mz_list_amplitude = mz_list[argrelmax(mz_list)]
            reservoir_data[f'mz_list_amplitude_{i}'] = mz_list_amplitude

        data = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in reservoir_data.items()])) 
        data.to_csv(f'{save_path}/ratio_{ratio}_{consuming_time}_{ac_current}_random_{random_index}.csv', index=False)
        print('save data successfully!')


if __name__ == '__main__':
    # multi-test
    ac_list = np.linspace(1, 100, 100, dtype=int)
    ratio_list = [0.5, 0.7, 0.8, 0.9]
    for ratio in ratio_list:
        with Pool(25) as pool:
            pool.starmap(
                esp_index_calculation,
                zip(itertools.repeat(50), itertools.repeat(1000), itertools.repeat(ratio), itertools.repeat(4e-9), ac_list))

    mtj_module.email_alert('Test results are ready !')

