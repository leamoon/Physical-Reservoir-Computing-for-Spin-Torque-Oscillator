import ipc_module
import mtj_module
import numpy as np
import pandas as pd

def ipc_target(length_input, task, ratio, delay, washtime, degree_ipc, max_delay_ipc):
    s_in, train_signal = mtj_module.real_time_generator(
        task=task, superposition_number=delay, length_signals=length_input+washtime, posibility_0=ratio
        )
    # print(s_in[:10], '\n',  train_signal[:10])
    reservoir_states = train_signal[washtime:].reshape(length_input, -1)
    # print(reservoir_states.shape)
    ipc_analyze = ipc_module.ipc(
        washtime=washtime, s_in=s_in, reservoir_states=reservoir_states, N_binomial=1, p_binomial=0.5,
        distribution_in='bernoulli', degree=degree_ipc, max_delay=max_delay_ipc, scale_factor=1.2
        )
    ipc_list = ipc_analyze.thresold()
    ipc_analyze.save_ipc()
    print(ipc_list)
    print('degree', degree_ipc, 'delay', max_delay_ipc, 'ipc', np.sum(ipc_list))


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
    # simple test for the restructure of ipc module
    # #####################################################################################
    ipc_target(length_input=10000, task='Delay', delay=1 ,ratio=0.5, washtime=5000, degree_ipc=1, max_delay_ipc=10)
