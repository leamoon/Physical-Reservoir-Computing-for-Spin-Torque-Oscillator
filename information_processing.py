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

def polynominal_calculation(reservoir_states, s_in, degree=1, max_delay=3, save_index=True, polynominal='legendre', sorrogate=False):
    # for reproduce the figures in paper.

    # generate the family of sets of degrees and delays
    global number_list
    data_dic = {} # save all data

    integrate_splitting(degree)
    n_list = [i.split(',') for i in number_list]
    number_list = []
    n_list.append([0]*max_delay)
    family_matrix = pd.concat(
        [pd.DataFrame({'{}'.format(index):labels}) for index,labels in enumerate(n_list)],axis=1
    ).fillna(0).values.T.astype(int)

    family_matrix = np.delete(family_matrix, -1, 0)
    total_family_matrix = family_matrix.copy()
    # prepare the degree, delay sets
    for index in range(family_matrix.shape[0]):
        all_iter_list = list(multiset_permutations(family_matrix[index], max_delay))
        all_iter_list.reverse()
        total_family_matrix = np.insert(total_family_matrix, index, np.matrix(all_iter_list), axis=0)

    total_family_matrix, idx = np.unique(total_family_matrix, axis=0, return_index=True)
    total_family_matrix = total_family_matrix[np.argsort(idx)]

    # generate the input matrix with different delay elements
    s_in_matrix = np.zeros((max_delay, len(s_in)))

    for delay_value in range(max_delay):
        # for shuffle sorrogate
        if sorrogate:
            s_in_matrix[delay_value, :] = s_in
        else:
            s_in_matrix[delay_value, :] = np.append(s_in[-(delay_value+1):], s_in[:-(delay_value+1)])

    # print(f's_in_matrix: \n{s_in_matrix}')
    # print('*****************************')
    polynominal_matrix = np.ones((total_family_matrix.shape[0], len(s_in)))

    # generate corresponding polynomial chaos
    for index_ipc in range(total_family_matrix.shape[0]):
        if polynominal == 'uniform':
            # legendre chaos
            polynomial = scipy.special.eval_legendre(total_family_matrix[index_ipc].T, s_in_matrix.T)
        elif polynominal == 'guassian':
            # hermite chaos
            polynomial = scipy.special.eval_hermite(total_family_matrix[index_ipc].T, s_in_matrix.T)

        elif polynominal == 'beta':
            alpha, beta = -0.25, -0.25
            polynomial = scipy.special.eval_jacobi(
                total_family_matrix[index_ipc].T, alpha, beta, s_in_matrix.T)

        elif polynominal == 'gamma':
            polynomial = scipy.special.eval_laguerre(total_family_matrix[index_ipc].T, s_in_matrix.T)

        polynomial_term = np.prod(polynomial, axis=1)
        # polynomial_term = polynomial_term / np.sqrt(np.dot(polynomial_term, polynomial_term))
        polynominal_matrix[index_ipc, :] = polynomial_term/np.linalg.norm(polynomial_term)
        # print('polynomial_term', polynomial_term, polynominal_matrix.shape)

        # data_dic[f'polynominal_{index_ipc}'] = polynominal_matrix[index_ipc, :]

    # calculate the ipc
    if reservoir_states.shape[0] == polynominal_matrix.shape[1]:
        index_reservoir_mean = reservoir_states.shape[1]
    else:
        index_reservoir_mean = reservoir_states.shape[0]
    for index_mean_reservoir in range(index_reservoir_mean):
        # print(np.mean(reservoir_states[:, index_mean_reservoir]))
        reservoir_states[:, index_mean_reservoir] = reservoir_states[:, index_mean_reservoir] - np.mean(reservoir_states[:, index_mean_reservoir])
        

    # print(f'x before svd', reservoir_states[0, :].T)
    # reservoir_states = reservoir_states.reshape(len(reservoir_states), 1)
    normalized_x, _, _ = np.linalg.svd(reservoir_states, full_matrices=False)
    # print('x after svd', normalized_x[0, :])

    # print('reservoir', normalized_x.shape, normalized_x[0, :])
    # print('**************************************************')
    # print('test', np.sum(np.dot(normalized_x.T, polynominal_matrix.T[:, 5])**2))

    c_list = (np.dot(normalized_x.T, polynominal_matrix.T))**2
    # print(f'c_list : {c_list}')
    # print(c_list.shape)
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

    data_dic['n_list'] = family_matrix_index
    data_dic['c_list'] = np.sum(c_list, axis=0)
    data_frame = pd.DataFrame(data_dic)
    
    if save_index:
        data_frame.to_csv(save_name)

    return data_frame

def second_example_one_layer_rc(length=10000, sigma=0.2, degree=1, max_delay=10, save_index=False, polynomial='uniform'):
    washout_time = 10000
    np.random.seed(0)
    if polynomial == 'uniform':
        # for legendre chaos
        s_in = np.random.uniform(-1, 1, length+washout_time)
    elif polynomial == 'guassian':
        # for Guassian chaos
        s_in = np.random.normal(loc=0, scale=1, size=length+washout_time)
        
    elif polynomial == 'beta':
        s_in = np.random.beta(0.25, 0.25, size=length+washout_time)*2-1

    elif polynomial == 'binomial':
        s_in = np.random.binomial(n=10, p=0.5, size=length+washout_time)

    elif polynomial == 'gamma':
        s_in = np.random.gamma(shape=1, scale=1, size=length+washout_time)
    
    print(f'input: {polynomial}')
    reservoir_state = 0
    reservoir_states = np.zeros((length+washout_time, 1))
    rho = 0.95
    for i in range(1, washout_time+length):
        reservoir_state = np.tanh(rho*reservoir_state + sigma*s_in[i-1])
        reservoir_states[i, 0] = reservoir_state
    
    reservoir_states = reservoir_states[washout_time:]
    s_in = s_in[washout_time:]
    # print('reservoir', reservoir_states)
    data_frame = polynominal_calculation(reservoir_states=reservoir_states, s_in=s_in, degree=degree, max_delay=max_delay, save_index=save_index, polynominal=polynomial)
    total_ipc_frame = {}
    degree_list = data_frame['n_list']
    c_list = data_frame['c_list']
    scale_factor = 1.2

    # update
    for i in track(range(200)):
        np.random.seed(i)
        random_input = np.random.permutation(s_in)
        data_frame = polynominal_calculation(
            reservoir_states=reservoir_states, s_in=random_input, 
            degree=degree, max_delay=max_delay, save_index=save_index, 
            polynominal=polynomial, sorrogate=False)
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
    total_ipc_frame.to_csv(f'sigma_{sigma}_degree_{degree}_delay_{max_delay}_{polynomial}.csv')

    data_frame = pd.DataFrame({'c_thresold_list': c_thresold_list, 'c_list': c_list})
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

def example_from_author_code(polynomial='legendre', degree=1, max_delay=1):
    ##### Parameters for esn #####
    N = 50      # Number of nodes
    Two = 10000 # Washout time
    T = 10000 # Time length except washout
    p = 0.5     # Sparsity for internal weight
    pin = 0.1   # Sparsity for input weight
    iota = 0.1  # Input intensity
    # Spectral radius
    rho = 0.1

    #Weight
    seed = 0
    np.random.seed(seed)
    win = (2*np.random.rand(N)-1) * (np.random.rand(N)<pin)
    w = (2*np.random.rand(N,N)-1) * (np.random.rand(N,N)<p)
    eig, _ = np.linalg.eig(w)
    w = w/np.max(np.abs(eig))

    ##### Input #####
    np.random.seed(0)
    zeta = 2*np.random.rand(Two+T)-1
    # print('zeta',zeta,zeta.shape)

    x = np.zeros((N,Two+T))
    for t in range(1,Two+T):
        x[:,t] = np.tanh(rho*w.dot(x[:,t-1])+iota*win*zeta[t-1])

    reservoir_states = x[:, Two:].T

    # print(reservoir_states.T, reservoir_states.shape)
    # print('****************************')
    s_in = zeta[Two:]
    data_frame = polynominal_calculation(reservoir_states=reservoir_states, s_in=s_in, degree=degree, max_delay=max_delay, save_index=False, polynominal=polynomial)
    total_ipc_frame = {}
    degree_list = data_frame['n_list']
    c_list = data_frame['c_list']
    scale_factor = 1.2
    # print(c_list)
    # sys.exit()

    # update
    for i in track(range(200)):
        np.random.seed(i)
        random_input = np.random.permutation(s_in)
        data_frame = polynominal_calculation(reservoir_states=reservoir_states, s_in=random_input, degree=degree, max_delay=max_delay, save_index=False, polynominal=polynomial)
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
    total_ipc_frame.to_csv(f'Example_degree_{degree}_delay_{max_delay}_{polynomial}.csv')

    data_frame = pd.DataFrame({'c_thresold_list': c_thresold_list, 'c_list': c_list})
    print(data_frame)
    print(f'degree = {degree}, c_total = {np.sum(c_thresold_list)}, Rank: {x.shape[0]}')

    return c_thresold_list, np.sum(c_thresold_list)

if __name__ == '__main__':
    # # test for first example in ipc calculation
    # s = np.random.normal(loc=0, scale=1, size=1000000)
    # plt.figure()
    # count, bins, ignored = plt.hist(s, 100, density=True)
    # plt.plot(bins, 1/(1 * np.sqrt(2 * np.pi)) *
    #            np.exp( - (bins - 0)**2 / (2 * 1**2) ),
    #      linewidth=2, color='r')
    # plt.show()

    # degree_delay_family = [[1, 500], [2, 100], [3, 50], [4, 10]]
    # # degree_delay_family = [[1, 100], [2, 100], [3, 50], [4, 10]]
    # # degree_delay_family = [[1, 100], [1, 200], [3, 20], [3, 50]]
    # sigma_list = np.round(np.linspace(0.2, 2, 10), 1)
    # # sigma_list = [1]
    # sigma_list = [0.2] 
    # # print(sigma_list)
    # # sys.exit()

    # for degree, max_delay in degree_delay_family:
    #     for sigma in sigma_list:
    #         _, ipc = second_example_one_layer_rc(length=10000, degree=degree, max_delay=max_delay, save_index=False, sigma=sigma, polynomial='legendre')
    
    # mtj_module.email_alert('hahaha, ipc calculation')
    # s_in = np.random.gamma(shape=1, scale=1, size=100000)
    # plt.figure()
    # count, bins, ignored = plt.hist(s_in, 100, density=True)
    # plt.show()

    delay_degree_list = [[1, 100], [2, 100], [3, 50], [4, 30], [5, 20], [6, 10], [7, 10], [8, 10]]
    sigma_list = np.round(np.linspace(0.2, 2, 10), 1)
    print(sigma_list)
    for degree, max_delay in delay_degree_list:
        for sigma in sigma_list:
            second_example_one_layer_rc(degree=degree, max_delay=max_delay, length=10000, polynomial='beta', sigma=sigma)

    sigma_list = np.round(np.linspace(0.1, 1, 10), 1)
    for degree, max_delay in delay_degree_list:
        for sigma in sigma_list:
            second_example_one_layer_rc(degree=degree, max_delay=max_delay, length=10000, polynomial='gamma', sigma=sigma)

    mtj_module.email_alert('reproduction beta!')