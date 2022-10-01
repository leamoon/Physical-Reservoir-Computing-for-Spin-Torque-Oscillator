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


np.random.seed(110)

def save_random_reservoir(length=10000, ratio=0.5, consuming_time=3e-9, ac_current=0.0):
    save_path='radom_input_data'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.exists(f'{save_path}/input_ratio_{ratio}_{consuming_time}_{ac_current}.csv'):
        print('random input files already exit')
        return 'random input files already exit'
    
    s_in = np.random.choice([0, 1], size=length, p=[ratio, 1-ratio])
    reservoir_data = {'s_in': s_in}

    initial_m = np.random.random(3)
    device = mtj_module.Mtj(initial_m[0], initial_m[1], initial_m[2])
    device.time_evolution(dc_amplitude=100, time_consumed=consuming_time)

    for i in track(range(len(s_in))):
        if s_in[i] == 0:
            _, _, mz_list, _, _ = device.time_evolution(dc_amplitude=100, time_consumed=consuming_time, ac_amplitude=ac_current)
        else:
            _, _, mz_list, _, _ = device.time_evolution(dc_amplitude=200, time_consumed=consuming_time, ac_amplitude=ac_current)

        mz_list_amplitude = mz_list[argrelmax(mz_list)]
        reservoir_data[f'mz_list_amplitude_{i}'] = mz_list_amplitude

    # print(reservoir_data)
    # save data

    data = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in reservoir_data.items()])) 
    data.to_csv(f'{save_path}/input_ratio_{ratio}_{consuming_time}_{ac_current}.csv')
    print('save data successfully!')
    mtj_module.email_alert(f'The {consuming_time} {ratio} task is finished.')


def efficient_training(save_path='radom_input_data', consuming_time=3e-9, ac_current=0.0):
    if not os.path.exists(save_path):
        sys.exit('No such directionary')
    nodes = [2, 5, 10, 16, 20, 30, 40, 50, 70, 90, 100]
    superposition_list = np.linspace(1, 10, 10, dtype=int)
    posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1)
    task_list = ['Delay', 'Parity']

    for ratio in track(posibility_list):
        # loading datas
        input_data_file = pd.read_csv(f'{save_path}/input_ratio_{ratio}_{consuming_time}.csv')
        s_in = input_data_file['s_in'].to_numpy()
        file_path = f'weight_evolution_{consuming_time}/weight_posibility_{ratio}'
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        for task in task_list:
            for node in nodes:
                for superposition in superposition_list:
                    if os.path.exists(f'{file_path}/STM_{task}_{superposition}_node_{node}_ac_0.0.npy'):
                        print('The weight aleardy existed')
                        continue
                        # return 'The weight aleardy existed'

                    if task == 'Delay':
                        train_signal = np.append(s_in[-int(superposition):], s_in[:-int(superposition)])

                    elif task == 'Parity':
                        train_signal = s_in
                        for super_value in range(1, superposition + 1):
                            temp_signal = np.append(s_in[-int(super_value):], s_in[:-int(super_value)])
                            train_signal = train_signal + temp_signal
                            train_signal[np.argwhere(train_signal == 2)] = 0

                    x_final_matrix = []
                    for index in range(len(s_in)):
                        # update weight
                        mz_amplitude = input_data_file[f'mz_list_amplitude_{index}']
                        mz_amplitude = mz_amplitude[~np.isnan(mz_amplitude)]
                        xp = np.linspace(1, len(mz_amplitude), len(mz_amplitude))
                        fp = mz_amplitude
                        sampling_x_values = np.linspace(1, len(mz_amplitude), node)
                        f = interp1d(xp, fp, kind='quadratic')
                        x_matrix1 = f(sampling_x_values)
                        x_matrix1 = np.append(x_matrix1.T, 1).reshape(-1, 1)
                        x_final_matrix.append(x_matrix1.T.tolist()[0])
                            
                    y_train_matrix = np.array(train_signal).reshape(1, len(train_signal))
                    x_final_matrix = np.asmatrix(x_final_matrix).T
                    temp_1 = np.dot(y_train_matrix, x_final_matrix.T)
                    weight_out = np.dot(temp_1, np.linalg.pinv(np.dot(x_final_matrix, x_final_matrix.T)))

                    y_test = np.dot(weight_out, x_final_matrix)
                    error_learning = np.var(np.array(train_signal) - np.array(y_test))
                    print(f'train error: {error_learning}, ratio: {ratio}, task: {task}')
                    
                    np.save(f'{file_path}/STM_{task}_{superposition}_node_{node}_ac_{ac_current}.npy', weight_out)

if __name__ == '__main__':
    time_list = [2e-9, 3e-9, 4e-9, 6e-9, 7e-9, 10e-9, 20e-9]
    time_list = [2e-9, 3e-9, 4e-9, 6e-9, 7e-9, 20e-9]
    ac_list = np.linspace(1, 100, 100, dtype=int)
    posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1) 
    for time in time_list:
        for posibility in posibility_list:
            with Pool() as pool:
                pool.starmap(save_random_reservoir, 
                            zip(itertools.repeat(10000), itertools.repeat(posibility), itertools.repeat(time), ac_list))

    # time_list = [2e-9, 3e-9, 4e-9, 6e-9, 7e-9, 10e-9, 20e-9]
    # with Pool() as pool:
    #     pool.starmap(efficient_training, zip(itertools.repeat('radom_input_data'), time_list))