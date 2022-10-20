import numpy as np
import os
import sys
import pandas as pd
from rich.progress import track
import scipy
from sympy.utilities.iterables import multiset_permutations

class ipc:
    def __init__(
        self, washtime, s_in, reservoir_states, distribution_in, degree, max_delay, 
        scale_factor, N_binomial=10, p_binomial=0.5
        ) -> None:
        self.washtime = washtime
        self.reservoir_states = reservoir_states
        self.s_in = s_in # the washtime should be smaller than length of s_in
        self.save_s_in = self.s_in
        self.polynomial = distribution_in
        self.number_list = [] # generating the degree delay family sets
        self.degree = degree
        self.max_delay = max_delay
        self.data_dic = {} # save all data
        self.scale_factor = scale_factor

        # parameters for polynomial
        self.N = N_binomial
        self.p = p_binomial

        # self.polynomial_match() # generate the poly_matrix
        self.delay_degree_generator() # generate the degree_delay sets

        # reduce the time-bias term from reservoir states
        # print(reservoir_states.shape)
        if self.reservoir_states.shape[0] == len(self.s_in) - self.washtime:
            index_reservoir_mean = self.reservoir_states.shape[1]
        else:
            index_reservoir_mean = self.reservoir_states.shape[0]
        for index_mean_reservoir in range(index_reservoir_mean):
            # print('mean_value', np.mean(reservoir_states[:, index_mean_reservoir]))
            self.reservoir_states[:, index_mean_reservoir] = self.reservoir_states[:, index_mean_reservoir] - np.mean(
                self.reservoir_states[:, index_mean_reservoir])
        
        self.svd_rc_states, _, _ = np.linalg.svd(self.reservoir_states, full_matrices=False)
        print('initialize ...')


    def polynomial_match(self):
        # generate the corresponding polynomial chaos
        self.degree_list = np.linspace(0, self.degree, self.degree+1, dtype=int)
        self.polynomial_matrix = np.ones((self.degree+1, len(self.s_in)))

        if self.polynomial == 'Uniform':
            self.polynomial_matrix = scipy.special.eval_legendre(self.degree_list.T, self.s_in)

        elif self.polynomial == 'Gaussian':
            self.polynomial_matrix = np.ones((self.degree+1, len(self.s_in)))
            self.polynomial_matrix[1] = self.s_in
            for n in range(1, self.degree):
                self.polynomial_matrix[n+1] = self.s_in*self.polynomial_matrix[n] - n*self.polynomial_matrix[n-1]

        elif self.polynomial == 'beta':
            alpha, beta = -0.25, -0.25
            self.polynomial_matrix = scipy.special.eval_jacobi(
                self.degree_list, alpha, beta, self.s_in)
        
        elif self.polynomial == 'gamma':
            a = 1
            L = self.polynomial_matrix
            L[1] = 1+a-self.s_in
            for n in range(1, self.degree):
                L[n+1] = ((2*n+1+a-self.s_in)*L[n]-(n+a)*L[n-1])/(n+1)
            self.polynomial_matrix = L
    
        elif self.polynomial == 'binomial':
            N, p = self.N, self.p
            if self.N != 1:
                K = self.polynomial_matrix
                K[1] = (1-self.s_in/(p*N))
                for n in range(1, self.degree):
                    K[n+1] = ((p*(N-n)+n*(1-p)-self.s_in)*K[n] - n*(1-p)*K[n-1] )/(p*(N-n))
                self.polynomial_matrix = K

        elif self.polynomial == 'bernoulli': # special calculation for this pattern
            # print(f'input: bernouli p = {self.p}')
            mean_s_in = np.mean(self.s_in)
            self.polynomial_matrix = np.zeros((2, len(self.s_in)))
            self.polynomial_matrix[1] = self.s_in - mean_s_in

        else:
            print('no such distrubition configuration')
            sys.exit('Input Configuration Error')

    def integrate_splitting(self, n, startnum=1, out=''):
        # subfunction to generate the degree_delay family sets     
        for i in range(startnum,n//2 + 1):
            outmp = out
            outmp += str(i) + ','
            self.integrate_splitting(n-i,i,outmp)
            
        if n == startnum:
            self.number_list.append(out + str(n))
            return

        self.integrate_splitting(n,n,out)

    def delay_degree_generator(self):
        # main function used to generate the total degree-delay sets
        self.number_list = [] # should be clear before use this parameter
        self.integrate_splitting(self.degree)
        n_list = [i.split(',') for i in self.number_list]
        # print(n_list)
        # sys.exit()
    
        n_list.append([0]*self.max_delay)
        family_matrix = pd.concat(
            [pd.DataFrame({'{}'.format(index):labels}) for index,labels in enumerate(n_list)],axis=1
        ).fillna(0).values.T.astype(int)

        family_matrix = np.delete(family_matrix, -1, 0)
        total_family_matrix = family_matrix.copy()
        # prepare the degree, delay sets
        for index in range(family_matrix.shape[0]):
            all_iter_list = list(multiset_permutations(family_matrix[index], self.max_delay))
            all_iter_list.reverse()
            total_family_matrix = np.insert(total_family_matrix, index, np.matrix(all_iter_list), axis=0)

        total_family_matrix, idx = np.unique(total_family_matrix, axis=0, return_index=True)
        total_family_matrix = total_family_matrix[np.argsort(idx)]

        # save the degree-delay sets
        self.degree_delay_sets = []
        self.delay_list = np.linspace(1, self.max_delay, self.max_delay, dtype=int)
        for index in range(total_family_matrix.shape[0]):
            degree_delay_unit = []
            for index_sub in range(total_family_matrix.shape[1]):
                # print('shape of degree_list', self.degree_list.shape)
                # print('shape of degree_list', total_family_matrix.shape)
                degree_delay_unit.append([total_family_matrix[index, index_sub], self.delay_list[index_sub]])
            self.degree_delay_sets.append(degree_delay_unit)

        self.total_family_matrix = total_family_matrix
        print('Loading the delay-degree sets')        

    def compute(self):
        #ipc calculation
        self.polynomial_match()
        self.c_list = np.zeros((1, self.total_family_matrix.shape[0]))
        for index_ipc in range(self.total_family_matrix.shape[0]):
            # print('index_famliy', total_family_matrix[index_ipc, :])
            single_row_polynomial = np.ones((1, len(self.s_in[self.washtime:])))
            for index_ipc_column in range(self.total_family_matrix.shape[1]):
                degree_value = self.total_family_matrix[index_ipc, index_ipc_column]
                if degree_value == 0:
                    continue
                delay_value = index_ipc_column+1
                polynomial_term = self.polynomial_matrix[degree_value, :]
                # print(f'polu_term: {polynomial_term}, degree{degree_value}, delay_value: {delay_value}')
                cor_temp = polynomial_term[self.washtime-delay_value: -delay_value].reshape(1, len(self.s_in[self.washtime:]))
                # print('cor_temp', cor_temp)
                single_row_polynomial = np.multiply(single_row_polynomial, cor_temp)
                # print(single_row_polynomial)
            single_row_polynomial = single_row_polynomial / np.linalg.norm(single_row_polynomial)
            # print(single_row_polynomial, 'norm')
            # print(single_row_polynomial.shape, normalized_x.shape)
            self.c_list[0, index_ipc] = np.sum((np.dot(self.svd_rc_states.T, single_row_polynomial.T))**2, axis=0)
        return self.c_list

    def thresold(self):
        self.origin_ipc = self.compute()
        self.total_ipc_frame = {}
        for i in track(range(200)):
            np.random.seed(i)
            # print('before', self.save_s_in)
            self.s_in = np.random.permutation(self.save_s_in)
            # print('after', self.s_in)
            self.compute()
            self.total_ipc_frame[f'c_list_{i}'] = self.c_list[0, :]
        
        # save all data
        self.total_ipc_frame = pd.DataFrame(self.total_ipc_frame)
        maximum_surrogate_values = self.total_ipc_frame.max(axis=1) * self.scale_factor
        self.c_thresold_list = []
        for i in range(len(maximum_surrogate_values)):
            if maximum_surrogate_values[i] > self.origin_ipc[0, i]:
                self.c_thresold_list.append(0)
            else:
                self.c_thresold_list.append(self.origin_ipc[0, i])
        # print(self.origin_ipc)
        # print('******************')
        # print(self.total_ipc_frame.max(axis=1))
        return self.c_thresold_list

    def save_ipc(self, path=None):
        if not path is None:
            save_path = path
        else:
            save_path = os.getcwd()
 
        self.total_ipc_frame.insert(self.total_ipc_frame.shape[1], 'c_list', self.origin_ipc[0, :])
        self.total_ipc_frame.insert(self.total_ipc_frame.shape[1], 'c_thresold_list', self.c_thresold_list)
        self.total_ipc_frame.insert(0, 'n_list', self.degree_delay_sets)

        summary_ipc_frame = pd.DataFrame({
            'c_thresold_list': self.c_thresold_list, 'degree_delay_sets': self.degree_delay_sets})

        # save
        self.total_ipc_frame.to_csv(f'{save_path}/test_degree_{self.degree}_delay_{self.max_delay}_{self.polynomial}.csv', index=False)
        summary_ipc_frame.to_csv(f'{save_path}/summary_test_degree_{self.degree}_delay_{self.max_delay}_{self.polynomial}.csv', index=False)
        print('data save successfully!')
