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

def polynominal_calculation_origin(reservoir_states, s_in, degree=1, max_delay=3, save_index=True, polynominal='legendre', sorrogate=False):
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
        polynominal_matrix[index_ipc, :] = polynomial_term/np.linalg.norm(polynomial_term)
  
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
    print(f'total ipc: {np.sum(c_list)}, ideal value: {normalized_x.shape[1]}')
    # sys.exit()

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

def polynominal_calculation(reservoir_states, s_in, washout=10000, degree=1, max_delay=3, polynominal='uniform'):
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

     # generate the SVD value of reservoir states
    if reservoir_states.shape[0] == len(s_in[washout:]):
        index_reservoir_mean = reservoir_states.shape[1]
    else:
        index_reservoir_mean = reservoir_states.shape[0]
    # print('index_mean', index_reservoir_mean)
    for index_mean_reservoir in range(index_reservoir_mean):
        # print(np.mean(reservoir_states[:, index_mean_reservoir]))
        reservoir_states[:, index_mean_reservoir] = reservoir_states[:, index_mean_reservoir] - np.mean(reservoir_states[:, index_mean_reservoir])
    normalized_x, _, _ = np.linalg.svd(reservoir_states, full_matrices=False)

    # generate the polynomial_matrix shape=(degree+1, len(s_in))
    degree_list = np.linspace(0, degree, degree+1, dtype=int).reshape(degree+1, 1)
    if polynominal == 'uniform':
        # legendre chaos
        polynomial_matrix = scipy.special.eval_legendre(degree_list, s_in.reshape(1, len(s_in)))
        # a,b = 0, 0
        # P = np.ones((degree+1, len(s_in)))
        # P[1] = ((a+b+2)*s_in+a-b)/2
        # for n in range(1,degree):
        #     A = (2*n+a+b+1)*((2*n+a+b)*(2*n+a+b+2)*s_in+(a**2)-(b**2))/(2*(n+1)*(n+a+b+1)*(2*n+a+b))
        #     B = -(n+a)*(n+b)*(2*n+a+b+2)/((n+1)*(n+a+b+1)*(2*n+a+b))
        #     P[n+1] = A*P[n]+B*P[n-1]
        # print(f'P: {P[-1, :]/np.linalg.norm(P[-1, :])}')
        # print(f'poly: {polynomial_matrix[-1, :]/np.linalg.norm(polynomial_matrix[-1, :])}')
        # print(P.shape, polynomial_matrix.shape)
        # sys.exit()

    elif polynominal == 'guassian':
        # hermite chaos
        # polynomial_matrix = scipy.special.eval_hermite(degree_list, s_in.reshape(1, len(s_in)))
        P = np.ones((degree+1, len(s_in)))
        P[1] = s_in
        for n in range(1, degree):
            P[n+1] = s_in*P[n] - n*P[n-1]
        # print(f'P: {P[-1, :]}')
        # print(f'poly: {polynomial_matrix[-1, :]}')
        # print(P.shape, polynomial_matrix.shape)
        # sys.exit()
        polynomial_matrix = P

    elif polynominal == 'beta':
        alpha, beta = -0.25, -0.25
        polynomial_matrix = scipy.special.eval_jacobi(
            degree_list, alpha, beta, s_in.reshape(1, len(s_in)))
        # a,b = alpha, beta
        # P = np.ones((degree+1, len(s_in)))
        # P[1] = ((a+b+2)*s_in+a-b)/2
        # for n in range(1,degree):
        #     A = (2*n+a+b+1)*((2*n+a+b)*(2*n+a+b+2)*s_in+(a**2)-(b**2))/(2*(n+1)*(n+a+b+1)*(2*n+a+b))
        #     B = -(n+a)*(n+b)*(2*n+a+b+2)/((n+1)*(n+a+b+1)*(2*n+a+b))
        #     P[n+1] = A*P[n]+B*P[n-1]
        # print(f'P: {P[-1, :]/np.linalg.norm(P[-1, :])}')
        # print(f'poly: {polynomial_matrix[-1, :]/np.linalg.norm(polynomial_matrix[-1, :])}')
        # print(P.shape, polynomial_matrix.shape)
        # sys.exit()

    elif polynominal == 'gamma':
        # polynomial_matrix = scipy.special.eval_laguerre(
        #     degree_list, s_in.reshape(1, len(s_in)))
        a = 1
        L = np.ones((degree+1, len(s_in)))
        L[1] = 1+a-s_in
        for n in range(1, degree):
            L[n+1] = ((2*n+1+a-s_in)*L[n]-(n+a)*L[n-1])/(n+1)
        polynomial_matrix = L
    
    elif polynominal == 'binomial':
        N, p = 10, 0.5
        K = np.ones((degree+1, len(s_in)))
        K[1] = (1-s_in/(p*N))
        for n in range(1, degree):
            K[n+1] = ((p*(N-n)+n*(1-p)-s_in)*K[n] - n*(1-p)*K[n-1] )/(p*(N-n))
        polynomial_matrix = K
   
    # generate corresponding polynomial chaos
    for i in range(polynomial_matrix.shape[0]):
        polynomial_matrix[i, :] = polynomial_matrix[i, :] / np.linalg.norm(polynomial_matrix[i, :])
    # print('input', s_in)
    # print('poly', polynomial_matrix)
    c_list = np.zeros((1, total_family_matrix.shape[0]))
    for index_ipc in range(total_family_matrix.shape[0]):
        # print('index_famliy', total_family_matrix[index_ipc, :])
        single_row_polynomial = np.ones((1, len(s_in[washout:])))
        for index_ipc_column in range(total_family_matrix.shape[1]):
            degree_value = total_family_matrix[index_ipc, index_ipc_column]
            if degree_value == 0:
                continue
            delay_value = index_ipc_column+1
            polynomial_term = polynomial_matrix[degree_value, :]
            # print(f'polu_term: {polynomial_term}, degree{degree_value}, delay_value: {delay_value}')
            cor_temp = polynomial_term[washout-delay_value: -delay_value].reshape(1, len(s_in[washout:]))
            # print('cor_temp', cor_temp)
            single_row_polynomial = np.multiply(single_row_polynomial, cor_temp)
            # print(single_row_polynomial)
        single_row_polynomial = single_row_polynomial / np.linalg.norm(single_row_polynomial)
        # print(single_row_polynomial, 'norm')
        # print(single_row_polynomial.shape, normalized_x.shape)
        c_list[0, index_ipc] = np.sum((np.dot(normalized_x.T, single_row_polynomial.T))**2, axis=0)
        # print(c_list[0, index_ipc])
        # sys.exit()
    # print(c_list)
    # print(c_list.shape, np.sum(c_list))
    # sys.exit()
    # save ipc data
    family_matrix_index = []
    for i in range(total_family_matrix.shape[0]):
        family_matrix_index.append(total_family_matrix[i, :])

    data_dic['n_list'] = family_matrix_index
    data_dic['c_list'] = c_list[0, :]
    data_frame = pd.DataFrame(data_dic)
    return data_frame

def second_example_one_layer_rc(length=10000, sigma=0.2, degree=1, max_delay=10, polynomial_index='uniform'):
    washout_time = 10000
    np.random.seed(0)
    if polynomial_index == 'uniform':
        # for legendre chaos
        s_in = np.random.uniform(-1, 1, length+washout_time)

    elif polynomial_index == 'guassian':
        # for Guassian chaos
        # s_in = np.random.normal(loc=0, scale=1, size=length+washout_time)
        s_in = np.random.randn(length+washout_time)
        
    elif polynomial_index == 'beta':
        s_in = np.random.beta(0.25, 0.25, size=length+washout_time)*2-1

    elif polynomial_index == 'binomial':
        s_in = np.random.binomial(n=10, p=0.5, size=length+washout_time)

    elif polynomial_index == 'gamma':
        s_in = np.random.gamma(shape=2, scale=1, size=length+washout_time)
    
    print(f'input: {polynomial_index}')
    reservoir_state = 0
    reservoir_states = np.zeros((length+washout_time, 1))
    rho = 0.95
    for i in range(1, washout_time+length):
        reservoir_state = np.tanh(rho*reservoir_state + sigma*s_in[i-1])
        reservoir_states[i, 0] = reservoir_state
    
    reservoir_states = reservoir_states[washout_time:]
    # s_in = s_in[washout_time:]
    # print('reservoir', reservoir_states)
    data_frame = polynominal_calculation(
        reservoir_states=reservoir_states, s_in=s_in, washout=washout_time, degree=degree, 
        max_delay=max_delay, polynominal=polynomial_index)
    total_ipc_frame = {}
    degree_list = data_frame['n_list']
    c_list = data_frame['c_list']
    scale_factor = 1.2

    # update
    for i in track(range(200)):
        np.random.seed(i)
        random_input = np.random.permutation(s_in)
        data_frame = polynominal_calculation(
            reservoir_states=reservoir_states, s_in=random_input, washout=washout_time,
            degree=degree, max_delay=max_delay,
            polynominal=polynomial_index)
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
    if not os.path.exists('ipc_data'):
        os.mkdir('ipc_data')    
    total_ipc_frame.to_csv(f'ipc_data/sigma_{sigma}_degree_{degree}_delay_{max_delay}_{polynomial_index}.csv')

    # data_frame = pd.DataFrame({'c_thresold_list': c_thresold_list, 'c_list': c_list})
    # print(data_frame)
    # print(
    #     f'degree = {degree}, c_total = {np.sum(c_thresold_list)}, Rank: {np.linalg.matrix_rank(reservoir_states)}, sigma = {sigma}')

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

def example_from_author_code(polynomial='uniform', degree=1, max_delay=1):
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

