import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def reset_ipc(degree, max_delay, path_data, sigma_list, distribution_inputs, scale_factor=1.2):
    if not os.path.exists(f'{path_data}/scale_{scale_factor}'):
        os.makedirs(f'{path_data}/scale_{scale_factor}')
    for sigma_index in range(len(sigma_list)):
        sigma = sigma_list[sigma_index]
        if os.path.exists(f'{path_data}/scale_{scale_factor}/Fast_sigma_{sigma}_degree_{degree}_delay_{delay}_{distribution_inputs}.csv'):
            continue

        df.drop(axis=1, columns=['n_list', 'c_thresold_list', 'c_list', 'Unnamed: 0'], inplace=True)
        # df.drop(df.iloc[0].index, axis=1, inplace=True)
        # print(list(df))

        maximum_surrogate_values = df.max(axis=1)*scale_factor
        # print(maximum_surrogate_values)
        c_thresold_list = []
        for i in range(len(maximum_surrogate_values)):
            if maximum_surrogate_values[i] > orgin_ipc[i]:
                c_thresold_list.append(0)
            else:
                c_thresold_list.append(orgin_ipc[i])

        new_thresold_values = pd.DataFrame({'c_thresold_list': c_thresold_list})
        # df.insert(0, 'n_list', degree_list)    
        new_thresold_values.to_csv(f'{path_data}/scale_{scale_factor}/Fast_sigma_{sigma}_degree_{degree}_delay_{delay}_{distribution_inputs}.csv')
        print(f'scale_factor{scale_factor} finish !')
        



if __name__ == '__main__':
    delay_degree_list = [[1, 100], [2, 80], [3, 50], [4, 30], [5, 10], [6, 10], [7, 10], [8, 10]]
    # delay_degree_list = [[1, 100], [2, 80], [3, 50], [4, 30], [5, 10]]
    # delay_degree_list = [[1, 100], [3, 50]]

    sigma_list = np.round(np.linspace(0.1, 1, 10), 1)
    sigma_list = np.round(np.linspace(0.2, 2, 10), 1)
    # sigma_list = np.round(np.linspace(0.05, 0.5, 10), 2)
    print(sigma_list)

    # data folder
    
    # os.chdir(path_name)
    
    distribution_inputs = 'uniform'
    scale_factor = 3
    # change the thresold value
    for degree, delay in delay_degree_list:
        reset_ipc(
            degree=degree, max_delay=delay, path_data='ipc_data', sigma_list=sigma_list, distribution_inputs=distribution_inputs, 
            scale_factor=scale_factor)

    path_name = f'ipc_data/scale_{scale_factor}'
    # path_name = 'ipc_data'
    os.chdir(path_name)
    plt.figure(f'one-rc example')
    sum_ipc_index = np.zeros((len(sigma_list), len(delay_degree_list)))

    for degree, delay in delay_degree_list:
        # ipc_list = []
        for sigma_index in range(len(sigma_list)):
            sigma = sigma_list[sigma_index]
            file_name = f'Fast_sigma_{sigma}_degree_{degree}_delay_{delay}_{distribution_inputs}.csv'
            df = pd.read_csv(file_name)
            # ipc_list.append(np.sum(df['c_thresold_list']))
            if np.sum(sum_ipc_index[sigma_index, :]) < 1:
                sum_ipc_index[sigma_index, degree-1] = np.sum(df['c_thresold_list'])
        
        print(f'sigma: {sigma_list}, \n ipc_total: {sum_ipc_index[:, degree-1]}')

        if degree == 1:
            plt.bar(sigma_list, sum_ipc_index[:, degree-1], edgecolor='black', width=sigma_list[0], tick_label=sigma_list, label='degree 1')
        else:
            bottom_value = np.sum(sum_ipc_index[:, :degree-1], axis=1)
            print('bottom value', bottom_value)
            plt.bar(sigma_list, sum_ipc_index[:, degree-1], edgecolor='black', width=sigma_list[0], tick_label=sigma_list, bottom=bottom_value, label=f'degree {degree}')
        
        
    plt.xlabel(r'$\sigma$', size=16)
    plt.ylabel(r'$C_{tot}$', size=16)
    plt.title(f'{distribution_inputs}', size=16)
    plt.legend()
    plt.show()

