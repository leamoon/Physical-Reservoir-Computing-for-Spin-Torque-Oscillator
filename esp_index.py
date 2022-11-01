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
        os.makedirs(save_path)

    for random_index in range(random_initial_states):
        if os.path.exists(f'{save_path}/ratio_{ratio}_{consuming_time}_{ac_current}_random_{random_index}.csv'):
            print('random input files already exit')
            s_in = pd.read_csv(f'{save_path}/ratio_{ratio}_{consuming_time}_{ac_current}_random_{random_index}.csv')['s_in']
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


def esp_analyze(random_initial_states=50, transition_time=500, ratio=0.5, consuming_time=4e-9, ac_current=0.0, node=20):
    # initialize the mtj module and m
    save_path = f'esp_data/{ratio}'
    data_file = f'{save_path}/ratio_{ratio}_{consuming_time}_{ac_current}_random_{random_initial_states-1}.csv'
    if not os.path.exists(data_file):
        sys.exit('no such file')

    dataframe = pd.read_csv(data_file)
    length_input = len(dataframe['s_in'])
    origin_s_in = dataframe['s_in'].tolist()
    original_orbit = np.zeros((length_input - transition_time, node)) # origin orbit
    difference_list_orbit = np.zeros((random_initial_states, 1)) # save the difference value between two orbits

    # generate the initial value of reservoir states 
    for row_index in range(length_input - transition_time):
        mz_amplitude = dataframe[f'mz_list_amplitude_{row_index+transition_time}']
        mz_amplitude = mz_amplitude[~np.isnan(mz_amplitude)]
        xp = np.linspace(1, len(mz_amplitude), len(mz_amplitude))
        fp = mz_amplitude
        sampling_x_values = np.linspace(1, len(mz_amplitude), node)
        f = interp1d(xp, fp, kind='quadratic')
        original_orbit[row_index, :] = f(sampling_x_values)

    # get the reservoir states with different random initial values
    for random_index in range(random_initial_states):
        data_file = f'{save_path}/ratio_{ratio}_{consuming_time}_{ac_current}_random_{random_index}.csv'
        if not os.path.exists(data_file):
            print('no such file')
            continue
        dataframe = pd.read_csv(data_file)
        random_initial_input = dataframe['s_in'].tolist()
        if origin_s_in != random_initial_input:
            print('error', random_index, 'ac_current', ac_current)
            difference_list_orbit[random_index, 0] = np.nan
        else:
            obrit_df_distance = np.zeros((length_input - transition_time, 1))

            for row_index in range(length_input - transition_time):
                mz_amplitude = dataframe[f'mz_list_amplitude_{row_index+transition_time}']
                mz_amplitude = mz_amplitude[~np.isnan(mz_amplitude)]
                xp = np.linspace(1, len(mz_amplitude), len(mz_amplitude))
                fp = mz_amplitude
                sampling_x_values = np.linspace(1, len(mz_amplitude), node)
                f = interp1d(xp, fp, kind='quadratic')
                obrit_df_distance[row_index, :] = np.linalg.norm(f(sampling_x_values) - original_orbit[row_index, :])
            
            difference_list_orbit[random_index, 0] = np.mean(obrit_df_distance[:, 0])
            # print('difference_list', difference_list_orbit.T)

    return np.nanmean(difference_list_orbit[:-1, 0])


if __name__ == '__main__':
    # multi-test
    ac_list = np.linspace(1, 100, 100, dtype=int)
    ratio_list = [0.5, 0.7, 0.8, 0.9, 0.6, 0.4, 0.3, 0.2, 0.1]
    for ratio in ratio_list:
        with Pool(15) as pool:
            pool.starmap(
                esp_index_calculation,
                zip(itertools.repeat(50), itertools.repeat(1000), itertools.repeat(ratio), itertools.repeat(4e-9), ac_list))

    mtj_module.email_alert('Test results are ready !')


    # test for eps_index_analyze
    # ac_list = np.linspace(1, 100, 100, dtype=int)
    # # ac_list = [1, 2, 3]
    # nodes = [16, 20, 30, 2, 5, 10]
    # for node in nodes:
    #     esp_value_matrix = np.zeros((len(ac_list), 1))
    #     for ac_value_index in track(range(len(ac_list))):
    #         esp_index_value = esp_analyze(ac_current=ac_list[ac_value_index], node=node, transition_time=500)
    #         esp_value_matrix[ac_value_index, 0] = esp_index_value
        
    #     data_frame = pd.DataFrame({'ac_current': ac_list, 'esp_list': esp_value_matrix[:, 0]})
    #     data_frame.to_csv(f'esp_node{node}_4e-9.csv', index=False)
    #     print(f'esp_node{node}_4e-9.csv save !')
