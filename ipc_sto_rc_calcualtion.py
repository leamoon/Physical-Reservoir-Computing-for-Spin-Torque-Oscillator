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

if __name__ == '__main__':
    # #####################################################################################
    # simple test for the restructure of ipc module
    # #####################################################################################
    # sigma = 0.05
    # washout_time = 10000
    # length = 10000
    # np.random.seed(0)
    # s_in = np.random.binomial(n=10, p=0.5, size=length+washout_time)
    # print(f'input: binomial')
    # reservoir_state = 0
    # reservoir_states = np.zeros((length+washout_time, 1))
    # rho = 0.95
    # for i in range(1, washout_time+length):
    #     reservoir_state = np.tanh(rho*reservoir_state + sigma*s_in[i-1])
    #     reservoir_states[i, 0] = reservoir_state
    
    # reservoir_states = reservoir_states[washout_time:]

    # delay_degree_list = [[1, 100], [2, 10], [3, 10], [4, 10], [5, 10], [6, 10], [7, 10], [8, 10]]
    # # delay_degree_list = [[7, 10], [8, 10]]
    # # delay_degree_list = [[1, 100]]
    # for degree, delay in delay_degree_list:
    #     ipc_class = ipc_module.ipc(
    #         washtime=washout_time, s_in=s_in, reservoir_states=reservoir_states, distribution_in='binomial', 
    #         degree=degree, max_delay=delay, scale_factor=3)
    #     ipc_list = ipc_class.thresold()
    #     ipc_class.save_ipc()
    #     # print(ipc_list)
    #     print('degree', degree, 'delay', delay, 'ipc', np.sum(ipc_list))

    # # df = pd.read_csv('summary_test_degree_8_delay_10_binomial.csv')
    # # c_list = df['c_thresold_list'].tolist()
    # # print(np.max(c_list))

    # #####################################################################################
    # calculation of ipc in STO-RC
    # #####################################################################################
    delay_degree_list = [[1, 50], [2, 10], [3, 10], [4, 10], [5, 10], [6, 10], [7, 10], [8, 10]]
    delay_degree_list = [[1, 10]]
    ac_list = np.linspace(0, 100, 101, dtype=int)
    posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1)
    # posibility_list = [0.1, 0.9]
    nodes = [100]
    for ratio in posibility_list:
        ipc_mtj(degree_delay_sets=delay_degree_list, ac_current=10, node=100, ratio=ratio, evolution_time=4e-9)
    # degree_delay_sets, ratio=0.5, evolution_time=4e-9, ac_current=0, node=20
    # for node in nodes:
    #     for posibility in posibility_list:
    #         with Pool(10) as pool:
    #             pool.starmap(ipc_mtj, 
    #                         zip(itertools.repeat(delay_degree_list), itertools.repeat(posibility), itertools.repeat(4e-9), ac_list, itertools.repeat(node)))

    # delay_degree_list = [[1, 100]]
    # for degree, max_delay in delay_degree_list:
    #     ipc_target(length_input=500, task='Parity', delay=3 ,ratio=0.5, washtime=500, degree_ipc=degree, max_delay_ipc=max_delay, scale_factor=1.2)

    # #####################################################################################
    # analyze the calculation of ipc in STO-RC
    # #####################################################################################
    # delay_degree_list = [[1, 200], [2, 100], [3, 20], [4, 20], [5, 10], [6, 10], [7, 10], [8, 10]]
    # ac_list = np.linspace(1, 100, 100, dtype=int)
    # posibility_list = np.round(np.linspace(0.1, 0.9, 9), 1)
    # nodes = [16, 20, 30, 10, 5]
    # node = 100
    # c_list_degree_rc = np.zeros((len(ac_list), len(delay_degree_list)))
    # for degree, delay in delay_degree_list:
    #     for ac_value_index in track(range(len(ac_list))):
    #         ac_value = ac_list[ac_value_index]
    #         file_name = f'ipc_data_mtj/4e-09/0.5/summary_test_degree_{degree}_delay_{delay}_bernoulli_ac_{ac_value}_node{node}.csv'
    #         df = pd.read_csv(file_name)
    #         c_list_degree_rc[ac_value_index, degree-1] = np.sum(df['c_thresold_list'])

    # print(c_list_degree_rc)
    # data = pd.DataFrame({
    #     'degree_1': c_list_degree_rc[:, 1],
    #     'degree_2': c_list_degree_rc[:, 2],
    #     'degree_3': c_list_degree_rc[:, 3],
    #     'degree_4': c_list_degree_rc[:, 4],
    #     })
    # data.to_csv('data_ipc_temp.csv', index=False)

    # data = pd.read_csv('data_ipc_temp.csv')
    # plt.figure(f'degree_1')
    # for i in range(1, 5):
        
    #     plt.plot(ac_list, data[f'degree_{i}'], label='degree 1')
    #     label_size = 16
    #     plt.xlabel(r'ac current', size=label_size)
    #     plt.ylabel(r'IPC', size=label_size)
    # plt.show()
