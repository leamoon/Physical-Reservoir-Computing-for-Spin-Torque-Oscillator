import mtj_module
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from rich.progress import track
from multiprocessing import Pool
import itertools
import scipy
import gc
from sympy.utilities.iterables import multiset_permutations

def first_example_ipc(length=10000):
    # length = 10000
    s_in = np.random.uniform(-1, 1, length)
    
    s_in_delay_1 = np.append(s_in[-1:], s_in[:-1])
    s_in_delay_2 = np.append(s_in[-2:], s_in[:-2])
    # print(s_in[: 5])
    # print(s_in_delay_2[:7])
    print(f'norm value of delay2 case: {np.linalg.norm((3*s_in_delay_2**2-1)/2)}')
    print(f'predicted value : {np.sqrt(length/5)}')

    x_state = s_in_delay_1 + s_in_delay_2**2

    new_x_state = (x_state - np.mean(x_state)).reshape(len(x_state), 1)

    print(f'real mean of x_state: {np.linalg.norm(new_x_state)}')
    print(f'predicted value of x_state: {np.sqrt(19*length/45)}')

    normalized_x, _, _ = np.linalg.svd(new_x_state, full_matrices=False)
    print(normalized_x.shape)
    # normalized_x_norm = np.repeat(np.linalg.norm(normalized_x, axis=1), normalized_x.shape[1], axis=0).reshape(normalized_x.shape)
    # normalized_x = np.divide(normalized_x, normalized_x_norm)

    polynominal = np.zeros((6, length))
    polynominal[0, :] = s_in_delay_1
    polynominal[1, :] = s_in_delay_2
    polynominal[2, :] = np.append(s_in[-3:], s_in[:-3])
    polynominal[3, :] = (3*s_in_delay_1**2-1)/2
    polynominal[4, :] = (3*s_in_delay_2**2-1)/2
    polynominal[5, :] = s_in_delay_1*s_in_delay_2

    # normalize
    polynominal[0, :] = polynominal[0, :]/np.linalg.norm(s_in_delay_1)
    polynominal[1, :] = polynominal[1, :]/np.linalg.norm(s_in_delay_2)
    polynominal[2, :] = polynominal[2, :]/np.linalg.norm(np.append(s_in[-3:], s_in[:-3]))
    polynominal[3, :] = polynominal[3, :]/np.linalg.norm((3*s_in_delay_1**2-1)/2)
    polynominal[4, :] = polynominal[4, :]/np.linalg.norm((3*s_in_delay_2**2-1)/2)
    polynominal[5, :] = polynominal[5, :]/np.linalg.norm(s_in_delay_1*s_in_delay_2)

    c_list = np.dot(normalized_x.T, polynominal.T)**2
    print(f'c_list : {c_list}')
    print(f'total ipc: {np.sum(c_list)}')

    ideal_value = [15/19, 0, 0, 0, 4/19, 0]
    print(f'ideal_value: {ideal_value}')

    plt.figure()
    x_label_list = np.linspace(1, 6, 6, dtype=int)
    plt.plot(x_label_list, ideal_value, label='ideal values')
    plt.plot(x_label_list, c_list[0, :], label='My results')
    plt.scatter(x_label_list, ideal_value)
    plt.legend()
    plt.ylabel(r'IPC ($C_i$)', size=16)
    plt.xlabel(r'$Degree: n_i$', size=16)
    plt.show()

def first_example_ipc_re(length=10000, degree=1, max_delay=3, save_index=False):
    # length = 10000
    s_in = np.random.uniform(-1, 1, length)
    
    s_in_delay_1 = np.append(s_in[-1:], s_in[:-1])
    s_in_delay_2 = np.append(s_in[-2:], s_in[:-2])

    x_state = s_in_delay_1 + s_in_delay_2**2
    data_frame = polynominal_calculation(reservoir_states=x_state, s_in=s_in, degree=degree, max_delay=max_delay, save_index=save_index)
    total_ipc_frame = {}
    degree_list = data_frame['n_list']
    c_list = data_frame['c_list']
    scale_factor = 1.2

    for i in range(200):
        np.random.seed(i)
        s_in = np.random.permutation(s_in)
        data_frame = polynominal_calculation(reservoir_states=x_state, s_in=s_in, degree=degree, max_delay=max_delay, save_index=save_index)
        total_ipc_frame[f'c_list_{i}'] = data_frame['c_list']

    # save all data
    total_ipc_frame = pd.DataFrame(total_ipc_frame)
    maximum_surrogate_values = total_ipc_frame.max(axis=1)*scale_factor
    c_thresold_list = []
    for i in range(len(maximum_surrogate_values)):
        if maximum_surrogate_values[i] > c_list[i]:
            c_thresold_list.append(0)
        else:
            c_thresold_list.append(c_list[i])

    total_ipc_frame.insert(total_ipc_frame.shape[1], 'c_list', c_list)
    total_ipc_frame.insert(total_ipc_frame.shape[1], 'c_thresold_list', c_thresold_list)
    total_ipc_frame.insert(0, 'n_list', degree_list)    
    total_ipc_frame.to_csv('test.csv')

    data_frame = pd.DataFrame({'c_thresold_list': c_thresold_list})
    print(data_frame)
    print(f'degree = {degree}, c_total = {np.sum(c_thresold_list)}, Rank: {np.linalg.matrix_rank(x_state)}')

    return c_thresold_list, np.sum(c_thresold_list)

def polynominal_calculation(reservoir_states, s_in, degree=1, max_delay=3, save_index=True):
    # for reproduce the figures in paper.

    # generate the family of sets of degrees and delays
    global number_list

    integrate_splitting(degree)
    n_list = [i.split(',') for i in number_list]
    number_list = []
    n_list.append([0]*max_delay)
    family_matrix = pd.concat(
        [pd.DataFrame({'{}'.format(index):labels}) for index,labels in enumerate(n_list)],axis=1
    ).fillna(0).values.T.astype(int)
    gc.collect() # release memory
    family_matrix = np.delete(family_matrix, -1, 0)
    total_family_matrix = family_matrix.copy()
    # prepare the degree, delay sets
    for index in range(family_matrix.shape[0]):
        all_iter_list = list(multiset_permutations(family_matrix[index], max_delay))
        total_family_matrix = np.insert(total_family_matrix, index+index*len(all_iter_list), np.matrix(all_iter_list), axis=0)

    total_family_matrix, idx = np.unique(total_family_matrix, axis=0, return_index=True)
    total_family_matrix = total_family_matrix[np.argsort(idx)]

    # generate the input matrix with different delay elements
    s_in_matrix = np.zeros((max_delay, len(s_in)))
    for delay_value in range(max_delay):
        s_in_matrix[delay_value, :] = np.append(s_in[-(delay_value+1):], s_in[:-(delay_value+1)])
    # print(f's_in_matrix: \n{s_in_matrix}')
    # print('*****************************')
    polynominal_matrix = np.zeros((total_family_matrix.shape[0], len(s_in)))

    # generate corresponding polynomial chaos
    for index_ipc in range(total_family_matrix.shape[0]):
        # legendre chaos
        polynomial = scipy.special.eval_legendre(total_family_matrix[index_ipc].T, s_in_matrix.T)
        polynominal_matrix[index_ipc, :] = np.prod(polynomial, axis=1) / np.linalg.norm(np.prod(polynomial, axis=1))

    # calculate the ipc
    reservoir_states = reservoir_states - np.mean(reservoir_states)
    reservoir_states = reservoir_states.reshape(len(reservoir_states), 1)
    normalized_x, _, _ = np.linalg.svd(reservoir_states, full_matrices=False)

    c_list = np.dot(normalized_x.T, polynominal_matrix.T)**2
    # print(f'c_list : {c_list}')
    # print(f'total ipc: {np.sum(c_list)}, ideal value: {normalized_x.shape[1]}')

    # save ipc data
    repeat_index = 0
    save_name = f'ipc_degree{degree}_maxdelay{max_delay}_{repeat_index}.csv'
    while os.path.exists(f'{save_name}'):
        repeat_index += 1
        save_name = f'ipc_degree{degree}_maxdelay{max_delay}_{repeat_index}.csv'
    family_matrix_index = []
    for i in range(total_family_matrix.shape[0]):
        family_matrix_index.append(total_family_matrix[i, :])

    data_frame = pd.DataFrame({'n_list': family_matrix_index, 'c_list': c_list[0]})
    if save_index:
        data_frame.to_csv(save_name)

    return data_frame


def second_example_one_layer_rc(length=10000, sigma=0.2, degree=1, max_delay=10, save_index=True):
    # for legendre chaos
    s_in = np.random.uniform(-1, 1, length)
    reservoir_state = 0
    reservoir_states = np.zeros((length, 1))
    rho = 0.95
    # for i in range(length):
    #     reservoir_state = np.tanh(rho*reservoir_state + sigma*s_in[i])
    for i in range(length):
        reservoir_state = np.tanh(rho*reservoir_state + sigma*s_in[i])
        reservoir_states[i, 0] = reservoir_state
    
    data_frame = polynominal_calculation(reservoir_states=reservoir_states, s_in=s_in, degree=degree, max_delay=max_delay, save_index=save_index)
    total_ipc_frame = {}
    degree_list = data_frame['n_list']
    c_list = data_frame['c_list']
    scale_factor = 1.2

    # update
    for i in track(range(100)):
        np.random.seed(i)
        random_input = np.random.permutation(s_in)
        data_frame = polynominal_calculation(reservoir_states=reservoir_states, s_in=random_input, degree=degree, max_delay=max_delay, save_index=save_index)
        total_ipc_frame[f'c_list_{i}'] = data_frame['c_list']
    
    # save all data
    total_ipc_frame = pd.DataFrame(total_ipc_frame)
    maximum_surrogate_values = total_ipc_frame.max(axis=1)*scale_factor
    c_thresold_list = []
    for i in range(len(maximum_surrogate_values)):
        if maximum_surrogate_values[i] > c_list[i]:
            c_thresold_list.append(0)
        else:
            c_thresold_list.append(c_list[i])

    total_ipc_frame.insert(total_ipc_frame.shape[1], 'c_list', c_list)
    total_ipc_frame.insert(total_ipc_frame.shape[1], 'c_thresold_list', c_thresold_list)
    total_ipc_frame.insert(0, 'n_list', degree_list)    
    total_ipc_frame.to_csv(f'sigma_{sigma}_degree_{degree}_delay_{max_delay}.csv')

    data_frame = pd.DataFrame({'c_thresold_list': c_thresold_list})
    print(data_frame)
    print(f'degree = {degree}, c_total = {np.sum(c_thresold_list)}, Rank: {np.linalg.matrix_rank(reservoir_states)}')

    return c_thresold_list, np.sum(c_thresold_list)
        


# need to be initialize before use the integrate_spliiting function
number_list = []
def integrate_splitting(n, startnum=1, out=''):
    
    global number_list
    for i in range(startnum,n//2 + 1):
        outmp = out
        outmp += str(i) + ','
        integrate_splitting(n-i,i,outmp)
        
    if n == startnum:
        number_list.append(out + str(n))
        return

    integrate_splitting(n,n,out)



if __name__ == '__main__':
    # # test for first example in ipc calculation
    # degree_list = [1, 2, 3]
    # ipc_sum = 0
    # for i in degree_list:
    #     _, ipc = first_example_ipc_re(degree=i, max_delay=4)
    #     ipc_sum += ipc
    # print(f'c_all: {ipc_sum}')

    degree_list = np.linspace(1, 4, 4, dtype=int)
    sigma_list = np.round(np.linspace(0.2, 2, 10), 1)
    for sigma in sigma_list:
        ipc_sum = 0
        for i in degree_list:
            _, ipc = second_example_one_layer_rc(length=10000, degree=i, max_delay=50, save_index=False, sigma=sigma)
            ipc_sum += ipc
            print(f'sigma: {sigma}, c_all: {ipc_sum}')

    # # test
    # a = np.linspace(0, 12, 12, dtype=int)
    # b = itertools.permutations(a, 12)
    # b_list = []
    # for i in b:
    #     b_list.append(i)
    # print(b)

