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

"""
This code is for detecting the best input configuration of physical reservoir computing system build by one single spin-torque oscillator
"""

np.random.seed(110)
initial_m = np.random.random(3)

def input_configuration_test(dc_current_list, ac_current, consuming_time, frequency):
    try:
        mz_amplitude_list = np.zeros((len(dc_current_list), 2))
        for i in track(range(len(dc_current_list))):
            dc_current = dc_current_list[i]
            device = mtj_module.Mtj(initial_m[0], initial_m[1], initial_m[2])
            device.time_evolution(dc_amplitude=dc_current, ac_amplitude=ac_current, time_consumed=2e-9, f_ac=frequency)
            mx_list, my_list, mz_list, _, m0 = device.time_evolution(dc_amplitude=dc_current, ac_amplitude=ac_current, 
                                                                     time_consumed=consuming_time, f_ac=frequency)
            # obtain the amplitude of mz
            if len(mz_list[argrelmax(mz_list)]) != 0:   
                mz_amplitude_list[i, 0] = np.max(mz_list[argrelmax(mz_list)])
            else:
                mz_amplitude_list[i, 0] = np.max(mz_list)

            if len(mz_list[argrelmin(mz_list)]) != 0:   
                mz_amplitude_list[i, 1] = np.min(mz_list[argrelmin(mz_list)])
            else:
                mz_amplitude_list[i, 1] = np.min(mz_list)

            df = pd.DataFrame({'dc_current': dc_current_list, 'mz_amplitude_max': mz_amplitude_list[:, 0], 'mz_amplitude_min': mz_amplitude_list[:, 1]})
            save_data_path = f'{os.getcwd()}/input_configuration_data'
            if not os.path.exists(save_data_path):
                os.makedirs(save_data_path)
            df.to_excel(f'{save_data_path}/input_configuraiton_{consuming_time*1e9}e-9.xlsx')
        print('saving successfully')

        # fig
        plt.figure()
        plt.plot(dc_current_list, mz_amplitude_list[:, 0])
        plt.scatter(dc_current_list, mz_amplitude_list[:, 0])
        plt.xlabel(r'DC current(Oe)', size=16)
        plt.ylabel(r'$M_{z}$ amplitude (a.u.)', size=16)
        plt.savefig(f'{save_data_path}/input_configuraiton_{consuming_time*1e9}e-9.png')
        plt.close('ALL')
        # plt.show()
    except Exception as error:
        print(f'dc_current: {dc_current},\n error: {error}')
        print(mz_list)
        df = pd.DataFrame({'mx_list': mx_list, 'my_list': my_list, 'mz_list': mz_list})
        save_data_path = f'{os.getcwd()}/error_data'
        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)
        df.to_excel(f'{save_data_path}/error_input_configuraiton_dc{dc_current}_{consuming_time*1e9}e-9.xlsx')


def  trajectories_detect(frequency, dc_current, ac_current, consuming_time=1e-9):
    # initial time
    # consuming_time = 1e-9
    initial_state = np.random.random(3)
    trace = mtj_module.Mtj(initial_state[0], initial_state[1], initial_state[2])
    trace.time_evolution(dc_amplitude=dc_current, ac_amplitude=ac_current, 
                         time_consumed=2e-9, f_ac=frequency)                

    m0 = trace.m
    mx_list_total, my_list_total, mz_list_total = [m0[0]], [m0[1]], [m0[2]]
    fig = plt.figure()
    # repeat
    while True:
        # evolution
        delta_t = consuming_time
        mx_list, my_list, mz_list, _, m0 = trace.time_evolution(
            dc_amplitude=dc_current, ac_amplitude=ac_current, time_consumed=delta_t, f_ac=frequency
            )
        show_trajectories = True
        if show_trajectories:
            mx_list_total = np.append(mx_list_total, mx_list)
            my_list_total = np.append(my_list_total, my_list)
            mz_list_total = np.append(mz_list_total, mz_list)
            fig.clf()
            
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122, projection='3d')

            # plt.subplot(2, 2, 1)
            # plt.plot(my_list_total, mz_list_total, label='normal')
            # plt.legend()
            # plt.subplot(2, 2, 2)
            # plt.plot(my_list_total_p, mz_list_total_p, label='perturbation')
            # plt.legend()
            # plt.subplot(2, 2, 3)
            # plt.plot(mz_list_total, c='green', label='normal')
            # plt.legend()
            # plt.subplot(2, 2, 4)
            # plt.plot(mz_list_total_p, c='blue', label='perturbation')
            # plt.legend()

            ax1.plot(mz_list_total)
            # ax = plt.axes(projection='3d')
            ax2.plot3D(mx_list_total, my_list_total, mz_list_total)
            ax2.scatter3D(mx_list_total[-1],my_list_total[-1],mz_list_total[-1], c='red', linewidth=2)
            null_z = [-1.1]*len(mx_list_total)
            null_y = [1.1]*len(mx_list_total)
            null_x = [-1.1]*len(mx_list_total)
            ax2.plot(mx_list_total, my_list_total, null_z, alpha=0.5, linewidth=0.8)
            ax2.plot(mx_list_total, null_y, mz_list_total, alpha=0.5, linewidth=0.8)
            ax2.plot(null_x, my_list_total, mz_list_total, alpha=0.5, linewidth=0.8)
            # ax.plot(mx_list_total,my_list_total,mz_list_total)

            plt.pause(0.1)
            plt.ioff()

def plot_multi_time(consuming_time_list, save_data_path):
    if not os.path.exists(save_data_path):
        print('no such path')
        sys.exit()
    
    plt.figure()
    plt.xlabel(r'DC current(Oe)', size=16)
    plt.ylabel(r'$M_{z}$ amplitude (a.u.)', size=16)

    for consuming_time in consuming_time_list:
        file_name = f'{save_data_path}/input_configuraiton_{consuming_time*1e9}e-9.xlsx'
        data = pd.read_excel(file_name)
        plt.plot(data['dc_current'], data['mz_amplitude_max'], label=f'{consuming_time*1e9}e-9s')

    plt.legend()
    plt.show()        


if __name__ == '__main__':
    dc_current_list = np.linspace(10, 3000, 2991, dtype=int)
    consuming_time_list = [5e-10, 7e-10, 1e-9, 2e-9]
    # dc_current_list = [204]
    # input_configuration_test(dc_current_list=dc_current_list, ac_current=0, consuming_time=5e-10, frequency=0)
    # trajectories_detect(frequency=0, dc_current=204, ac_current=0, consuming_time=5e-10)
    # with Pool() as pool:
    #     pool.starmap(input_configuration_test, 
    #                  zip(itertools.repeat(dc_current_list), itertools.repeat(0), 
    #                      consuming_time_list, itertools.repeat(0)))

    plot_multi_time(consuming_time_list=consuming_time_list, save_data_path='input_configuration_data')