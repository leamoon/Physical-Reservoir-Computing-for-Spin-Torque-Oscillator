import ipc_module
import mtj_module
import numpy as np
import pandas as pd
import os
import sys
from scipy.interpolate import interp1d
from multiprocessing import Pool
import itertools
from rich.progress import track
import matplotlib.pyplot as plt

def ipc_target(length_input, task, ratio, delay, washtime, degree_ipc, max_delay_ipc, scale_factor=3):
    s_in, train_signal = mtj_module.real_time_generator(
        task=task, superposition_number=delay, length_signals=length_input+washtime, posibility_0=ratio
        )
    # print(s_in[:10], '\n',  train_signal[:10])
    reservoir_states = train_signal[washtime:].reshape(length_input, -1)
    # print(reservoir_states.shape)
    ipc_analyze = ipc_module.ipc(
        washtime=washtime, s_in=s_in, reservoir_states=reservoir_states, N_binomial=1, p_binomial=0.5,
        distribution_in='bernoulli', degree=degree_ipc, max_delay=max_delay_ipc, scale_factor=scale_factor
        )
    ipc_list = ipc_analyze.thresold()
    ipc_analyze.save_ipc()

    # info about the max and min
    ipc_temp = []
    for i in ipc_list:
        if i != 0:
            ipc_temp.append(i)    
    print('max', np.max(ipc_list))
    if len(ipc_temp) != 0:
        print('min', np.min(ipc_temp))
    else:
        print('no min value')
    print('degree', degree_ipc, 'delay', max_delay_ipc, 'ipc', np.sum(ipc_list))

def ipc_mtj(degree_delay_sets, ratio=0.5, evolution_time=4e-9, ac_current=0, node=20):

    file_name = f'radom_input_data/input_ratio_{ratio}_{evolution_time}_{ac_current}.csv'
    if not os.path.exists(file_name):
        return f'no such file: {file_name}'
    input_data_file = pd.read_csv(file_name)
    s_in = input_data_file['s_in']
    washtime = int(len(s_in)/10)
    reservoir_states = np.zeros(((len(s_in) - washtime), node))

    # construct reservoir states
    for row_index in range(len(s_in) - washtime):
        mz_amplitude = input_data_file[f'mz_list_amplitude_{row_index+washtime}']
        mz_amplitude = mz_amplitude[~np.isnan(mz_amplitude)]
        xp = np.linspace(1, len(mz_amplitude), len(mz_amplitude))
        fp = mz_amplitude
        sampling_x_values = np.linspace(1, len(mz_amplitude), node)
        f = interp1d(xp, fp, kind='quadratic')
        reservoir_states[row_index, :] = f(sampling_x_values)
    print('rank of reservoir', np.linalg.matrix_rank(reservoir_states))
    # # calculate ipc
    # for degree_ipc, max_delay_ipc in degree_delay_sets:
    #     ipc_analyze = ipc_module.ipc(
    #         washtime=washtime, s_in=s_in, reservoir_states=reservoir_states, N_binomial=1, p_binomial=ratio,
    #         distribution_in='bernoulli', degree=degree_ipc, max_delay=max_delay_ipc, scale_factor=3
    #         )
    #     ipc_list = ipc_analyze.thresold()
    #     ipc_analyze.save_ipc(path=f'ipc_data_mtj/{evolution_time}/{ratio}', remark=f'ac_{ac_current}_node{node}')
    #     # info about the max and min
    #     ipc_temp = []
    #     for i in ipc_list:
    #         if i != 0:
    #             ipc_temp.append(i)    
    #     print('max', np.max(ipc_list))
    #     if len(ipc_temp) != 0:
    #         print('min', np.min(ipc_temp))
    #     else:
    #         print('no min value')
    #     # print('max', np.max(ipc_list), 'min', np.min(ipc_list))
    #     print('ac', ac_current, 'degree', degree_ipc, 'delay', max_delay_ipc, 'ipc', np.sum(ipc_list))
    return 0
