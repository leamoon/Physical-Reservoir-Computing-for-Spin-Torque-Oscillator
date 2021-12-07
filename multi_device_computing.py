import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import switch_mtj as s_mtj
import edge_esn_oscillator as o_mtj
from rich.progress import track
from scipy.interpolate import interp1d


def multi_system(length_int=100, node_size=20, task='Delay', superposition=0, recover_weight=False, ac_amplitude=0.0,
                 save_index=True, visual_process=True):
    # hyper parameters
    file_path = 'weight_matrix_multi_device'
    positive_dc_current_oscillator = 200
    negative_dc_current_oscillator = 100
    positive_dc_current_switch, negative_dc_current_switch = 200, -200
    time_consume_oscillator = 4e-9
    time_consume_switch = 8e-9
    f_ac = 32e9

    # build two reservoir physical devices
    switch_device = s_mtj.Mtj()
    oscillator_device = o_mtj.Mtj()

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if os.path.exists(f'{file_path}/STM_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy'):
        weight_out_stm = np.load(f'{file_path}/STM_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy')
        print('###############################################')
        print(f'{file_path}/STM_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy already exists !')
        print('###############################################')

        if not recover_weight:
            return 0
        else:
            print('NO MATTER to retrain again !')

    else:
        # think about bias term
        weight_out_stm = np.random.randint(-1, 2, (1, node_size * 2 + 1))
        print('\r weight matrix of STM created successfully !', end='', flush=True)

    # it seems to build a function better

    s_in, train_signal = o_mtj.real_time_generator(task=f'{task}', superposition_number=superposition,
                                                   length_signals=length_int)
    # pre training
    for pre_length in range(5):
        dc_current_list = [positive_dc_current_oscillator, negative_dc_current_oscillator]
        dc_current_value = dc_current_list[np.random.randint(0, 2, 1)[0]]
        oscillator_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                         time_consumed=time_consume_oscillator, f_ac=f_ac)
        dc_current_list = [positive_dc_current_switch, negative_dc_current_switch]
        dc_current_value = dc_current_list[np.random.randint(0, 2, 1)[0]]
        switch_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                     time_consumed=time_consume_switch)

    trace_mz, y_out_list, x_update_matrix = [], [], []

    for i in track(range(len(s_in))):
        if s_in[i] == 1:
            dc_current_value = positive_dc_current_oscillator
            _, _, mz_list, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_oscillator,
                                                                       f_ac=f_ac)
            dc_current_value = positive_dc_current_switch
            mx_list, _, _, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_switch)

        else:
            dc_current_value = negative_dc_current_oscillator
            _, _, mz_list, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_oscillator,
                                                                       f_ac=f_ac)
            dc_current_value = negative_dc_current_switch
            mx_list, _, _, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_switch)

        # for oscillator
        try:
            mz_list_all, t_list_whole = [], []
            if 'extreme_high' not in locals().keys():
                extreme_high, extreme_low = [], []
            else:
                extreme_high, extreme_low = [extreme_high[-1]], [extreme_low[-1]]

            mz_list, t_list_o = list(mz_list), list(t_list_o)

            for i1 in range(len(mz_list)):
                if i1 != 0 and i1 != len(mz_list) - 1:
                    if mz_list[i1] > mz_list[i1 - 1] and mz_list[i1] > mz_list[i1 + 1]:
                        extreme_high.append(mz_list[i1])
                    if mz_list[i1] < mz_list[i1 - 1] and mz_list[i1] < mz_list[i1 + 1]:
                        extreme_low.append(mz_list[i1])

            length_extreme = min(len(extreme_low), len(extreme_high))
            for i2 in range(length_extreme):
                mz_list_all.append(extreme_high[i2])

            xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
            fp = mz_list_all
            sampling_x_values = np.linspace(1, len(mz_list_all), node_size)
            f = interp1d(xp, fp, kind='quadratic')
            x_matrix_o = f(sampling_x_values)
            trace_mz = trace_mz + list(x_matrix_o)

        except Exception as ErrorMessage:
            print('error in building oscillator reservoir')
            sys.exit(ErrorMessage)

        # for switch device
        try:
            index_sample = np.linspace(0, len(mx_list) - 1, node_size, dtype=int, endpoint=False)
            x_matrix_s = np.array([mx_list[index_value] for index_value in index_sample])
            trace_mz = trace_mz + list(x_matrix_s)

        except Exception as ErrorMessage:
            print('error in building switch_device reservoir')
            sys.exit(ErrorMessage)

        # calculate output and update weights
        try:
            x_matrix = np.vstack([x_matrix_s, x_matrix_o])
            x_matrix = np.append(x_matrix.T, 1).reshape(-1, 1)
            y_out = np.dot(weight_out_stm, x_matrix)
            y_out_list.append(y_out[0, 0])
            x_update_matrix.append(x_matrix.T.tolist()[0])

        except ValueError as error:
            print('----------------------------------------')
            print('error from readout layer: {}'.format(error))
            print('Please check for your weight file at {}'.format(file_path))
            print('________________________________________')
            return 0

    # update weight
    y_train_matrix = np.array(train_signal).reshape(1, len(train_signal))
    x_update_matrix = np.asmatrix(x_update_matrix).T
    weight_out_stm = np.dot(y_train_matrix, np.linalg.pinv(x_update_matrix))

    y_test = np.dot(weight_out_stm, x_update_matrix)
    print('train result:{}'.format(y_test))

    # calculate the error -> mean square error
    error_learning = np.var(np.array(train_signal) - np.array(y_test))
    print('##################################################################')
    print('error:{}'.format(error_learning))
    print('Trained successfully !')
    print('##################################################################')

    # save weight matrix as .npy files
    if save_index:
        np.save(f'{file_path}/STM_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy', weight_out_stm)
        print('Saved weight matrix file')

    # visualization of magnetization
    if visual_process:
        plt.figure('Trace of mz')
        plt.title('reservoirs states')
        plt.xlabel('Time interval')
        plt.ylabel(r'$m_z$')
        t1 = np.linspace(0, len(trace_mz) - 1, len(trace_mz))
        plt.scatter(t1, trace_mz)

        plt.figure('input')
        plt.title('input signals')
        plt.plot(s_in)
        plt.ylabel('inputs')
        plt.xlabel('Time')
        plt.show()


def multi_system_test(length_int=10, node_size=20, task='Delay', superposition=0, ac_amplitude=0.0,
                      visual_process=True):
    # hyper parameters
    file_path = 'weight_matrix_multi_device'
    positive_dc_current_oscillator = 200
    negative_dc_current_oscillator = 100
    positive_dc_current_switch, negative_dc_current_switch = 200, -200
    time_consume_oscillator = 4e-9
    time_consume_switch = 8e-9
    f_ac = 32e9

    # build two reservoir physical devices
    switch_device = s_mtj.Mtj()
    oscillator_device = o_mtj.Mtj()

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if os.path.exists(f'{file_path}/STM_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy'):
        weight_out_stm = np.load(f'{file_path}/STM_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy')
        print('###############################################')
        print(f'{file_path}/STM_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy already exists !')
        print('###############################################')

    else:
        sys.exit('no valid data')

    # it seems to build a function better

    s_in, train_signal = o_mtj.real_time_generator(task=f'{task}', superposition_number=superposition,
                                                   length_signals=length_int)
    # pre training
    for pre_length in range(5):
        dc_current_list = [positive_dc_current_oscillator, negative_dc_current_oscillator]
        dc_current_value = dc_current_list[np.random.randint(0, 2, 1)[0]]
        oscillator_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                         time_consumed=time_consume_oscillator, f_ac=f_ac)
        dc_current_list = [positive_dc_current_switch, negative_dc_current_switch]
        dc_current_value = dc_current_list[np.random.randint(0, 2, 1)[0]]
        switch_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                     time_consumed=time_consume_switch)

    trace_mz, y_out_list, x_update_matrix = [], [], []

    for i in track(range(len(s_in))):
        if s_in[i] == 1:
            dc_current_value = positive_dc_current_oscillator
            _, _, mz_list, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_oscillator,
                                                                       f_ac=f_ac)
            dc_current_value = positive_dc_current_switch
            mx_list, _, _, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_switch)

        else:
            dc_current_value = negative_dc_current_oscillator
            _, _, mz_list, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_oscillator,
                                                                       f_ac=f_ac)
            dc_current_value = negative_dc_current_switch
            mx_list, _, _, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_switch)

        # for oscillator
        try:
            mz_list_all, t_list_whole = [], []
            if 'extreme_high' not in locals().keys():
                extreme_high, extreme_low = [], []
            else:
                extreme_high, extreme_low = [extreme_high[-1]], [extreme_low[-1]]

            mz_list, t_list_o = list(mz_list), list(t_list_o)

            for i1 in range(len(mz_list)):
                if i1 != 0 and i1 != len(mz_list) - 1:
                    if mz_list[i1] > mz_list[i1 - 1] and mz_list[i1] > mz_list[i1 + 1]:
                        extreme_high.append(mz_list[i1])
                    if mz_list[i1] < mz_list[i1 - 1] and mz_list[i1] < mz_list[i1 + 1]:
                        extreme_low.append(mz_list[i1])

            length_extreme = min(len(extreme_low), len(extreme_high))
            for i2 in range(length_extreme):
                mz_list_all.append(extreme_high[i2])

            xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
            fp = mz_list_all
            sampling_x_values = np.linspace(1, len(mz_list_all), node_size)
            f = interp1d(xp, fp, kind='quadratic')
            x_matrix_o = f(sampling_x_values)
            trace_mz = trace_mz + list(x_matrix_o)

        except Exception as ErrorMessage:
            print('error in building oscillator reservoir')
            sys.exit(ErrorMessage)

        # for switch device
        try:
            index_sample = np.linspace(0, len(mx_list) - 1, node_size, dtype=int, endpoint=False)
            x_matrix_s = np.array([mx_list[index_value] for index_value in index_sample])
            trace_mz = trace_mz + list(x_matrix_s)

        except Exception as ErrorMessage:
            print('error in building switch_device reservoir')
            sys.exit(ErrorMessage)

        # calculate output and update weights
        try:
            x_matrix = np.vstack([x_matrix_s, x_matrix_o])
            x_matrix = np.append(x_matrix.T, 1).reshape(-1, 1)
            y_out = np.dot(weight_out_stm, x_matrix)
            y_out_list.append(y_out[0, 0])
            x_update_matrix.append(x_matrix.T.tolist()[0])

        except ValueError as error:
            print('----------------------------------------')
            print('error from readout layer: {}'.format(error))
            print('Please check for your weight file at {}'.format(file_path))
            print('________________________________________')
            return 0

    # update weight
    y_train_matrix = np.array(train_signal).reshape(1, len(train_signal))
    x_update_matrix = np.asmatrix(x_update_matrix).T
    weight_out_stm = np.dot(y_train_matrix, np.linalg.pinv(x_update_matrix))

    y_test = np.dot(weight_out_stm, x_update_matrix)
    print('train result:{}'.format(y_test))

    # calculate the error -> mean square error
    error_learning = np.var(np.array(train_signal) - np.array(y_test))
    capacity = pow(np.corrcoef(y_out_list[superposition:], train_signal[superposition:])[0, 1], 2)
    print('##################################################################')
    print('error:{}'.format(error_learning))
    print('Trained successfully !')
    print('##################################################################')

    # visualization of magnetization
    if visual_process:
        plt.figure('Test results')
        plt.plot(train_signal[superposition:], c='blue', label='target')
        plt.plot(y_out_list[superposition:], c='green', label='module')
        plt.ylabel('signals')
        plt.xlabel('Time')
        plt.legend()

        plt.figure('Comparison')
        plt.subplot(2, 1, 1)
        plt.title('Mean square error : {}'.format(error_learning))
        plt.plot(train_signal[superposition:], c='blue', label='target output')
        plt.legend()
        plt.ylabel('signals')
        plt.xlabel('Time')

        plt.subplot(2, 1, 2)
        plt.plot(y_out_list[superposition:], c='red', label='actual output')
        # plt.title('Mean square error : {}'.format(error_learning))
        plt.legend()
        plt.ylabel('signals')
        plt.xlabel('Time')
        plt.show()

    return capacity


def multi_system_oscillator(length_int=100, node_size=20, task='Delay', superposition=0, recover_weight=False,
                            ac_amplitude=0.0,
                            save_index=True, visual_process=True):
    # hyper parameters
    file_path = 'weight_matrix_multi_device'
    positive_dc_current_oscillator = 200
    negative_dc_current_oscillator = 100
    time_consume_oscillator = 4e-9
    f_ac = 32e9

    # build two reservoir physical devices
    switch_device = o_mtj.Mtj()
    oscillator_device = o_mtj.Mtj()

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if os.path.exists(f'{file_path}/Oscillator_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy'):
        weight_out_stm = np.load(
            f'{file_path}/Oscillator_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy')
        print('###############################################')
        print(f'{file_path}/Oscillator_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy already exists !')
        print('###############################################')

        if not recover_weight:
            return 0
        else:
            print('NO MATTER to retrain again !')

    else:
        # think about bias term
        weight_out_stm = np.random.randint(-1, 2, (1, node_size * 2 + 1))
        print('\r weight matrix of STM created successfully !', end='', flush=True)

    # it seems to build a function better

    s_in, train_signal = o_mtj.real_time_generator(task=f'{task}', superposition_number=superposition,
                                                   length_signals=length_int)
    # pre training
    for pre_length in range(5):
        dc_current_list = [positive_dc_current_oscillator, negative_dc_current_oscillator]
        dc_current_value = dc_current_list[np.random.randint(0, 2, 1)[0]]
        oscillator_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                         time_consumed=time_consume_oscillator, f_ac=f_ac)
        dc_current_value = dc_current_list[np.random.randint(0, 2, 1)[0]]
        switch_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                     time_consumed=time_consume_oscillator)

    trace_mz, y_out_list, x_update_matrix = [], [], []

    for i in track(range(len(s_in))):
        if s_in[i] == 1:
            dc_current_value = positive_dc_current_oscillator
            _, _, mz_list, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_oscillator,
                                                                       f_ac=f_ac)
            _, _, mx_list, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_oscillator)

        else:
            dc_current_value = negative_dc_current_oscillator
            _, _, mz_list, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_oscillator,
                                                                       f_ac=f_ac)
            _, _, mx_list, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_oscillator)

        # for oscillator
        try:
            mz_list_all, t_list_whole = [], []
            if 'extreme_high' not in locals().keys():
                extreme_high, extreme_low = [], []
            else:
                extreme_high, extreme_low = [extreme_high[-1]], [extreme_low[-1]]

            mz_list, t_list_o = list(mz_list), list(t_list_o)

            for i1 in range(len(mz_list)):
                if i1 != 0 and i1 != len(mz_list) - 1:
                    if mz_list[i1] > mz_list[i1 - 1] and mz_list[i1] > mz_list[i1 + 1]:
                        extreme_high.append(mz_list[i1])
                    if mz_list[i1] < mz_list[i1 - 1] and mz_list[i1] < mz_list[i1 + 1]:
                        extreme_low.append(mz_list[i1])

            length_extreme = min(len(extreme_low), len(extreme_high))
            for i2 in range(length_extreme):
                mz_list_all.append(extreme_high[i2])

            xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
            fp = mz_list_all
            sampling_x_values = np.linspace(1, len(mz_list_all), node_size)
            f = interp1d(xp, fp, kind='quadratic')
            x_matrix_o = f(sampling_x_values)
            trace_mz = trace_mz + list(x_matrix_o)

        except Exception as ErrorMessage:
            print('error in building oscillator reservoir')
            sys.exit(ErrorMessage)

        # for switch device
        try:
            mz_list_all, t_list_whole = [], []
            if 'extreme_high' not in locals().keys():
                extreme_high, extreme_low = [], []
            else:
                extreme_high, extreme_low = [extreme_high[-1]], [extreme_low[-1]]

            mx_list, t_list_o = list(mx_list), list(t_list_o)

            for i1 in range(len(mx_list)):
                if i1 != 0 and i1 != len(mx_list) - 1:
                    if mx_list[i1] > mx_list[i1 - 1] and mx_list[i1] > mx_list[i1 + 1]:
                        extreme_high.append(mx_list[i1])
                    if mx_list[i1] < mx_list[i1 - 1] and mx_list[i1] < mx_list[i1 + 1]:
                        extreme_low.append(mx_list[i1])

            length_extreme = min(len(extreme_low), len(extreme_high))
            for i2 in range(length_extreme):
                mz_list_all.append(extreme_high[i2])

            xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
            fp = mz_list_all
            sampling_x_values = np.linspace(1, len(mz_list_all), node_size)
            f = interp1d(xp, fp, kind='quadratic')
            x_matrix_s = f(sampling_x_values)
            trace_mz = trace_mz + list(x_matrix_s)

        except Exception as ErrorMessage:
            print('error in building switch_device reservoir')
            sys.exit(ErrorMessage)

        # calculate output and update weights
        try:
            x_matrix = np.vstack([x_matrix_s, x_matrix_o])
            x_matrix = np.append(x_matrix.T, 1).reshape(-1, 1)
            y_out = np.dot(weight_out_stm, x_matrix)
            y_out_list.append(y_out[0, 0])
            x_update_matrix.append(x_matrix.T.tolist()[0])

        except ValueError as error:
            print('----------------------------------------')
            print('error from readout layer: {}'.format(error))
            print('Please check for your weight file at {}'.format(file_path))
            print('________________________________________')
            return 0

    # update weight
    y_train_matrix = np.array(train_signal).reshape(1, len(train_signal))
    x_update_matrix = np.asmatrix(x_update_matrix).T
    weight_out_stm = np.dot(y_train_matrix, np.linalg.pinv(x_update_matrix))

    y_test = np.dot(weight_out_stm, x_update_matrix)
    print('train result:{}'.format(y_test))

    # calculate the error -> mean square error
    error_learning = np.var(np.array(train_signal) - np.array(y_test))
    print('##################################################################')
    print('error:{}'.format(error_learning))
    print('Trained successfully !')
    print('##################################################################')

    # save weight matrix as .npy files
    if save_index:
        np.save(f'{file_path}/Oscillator_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy', weight_out_stm)
        print('Saved weight matrix file')

    # visualization of magnetization
    if visual_process:
        plt.figure('Trace of mz')
        plt.title('reservoirs states')
        plt.xlabel('Time interval')
        plt.ylabel(r'$m_z$')
        t1 = np.linspace(0, len(trace_mz) - 1, len(trace_mz))
        plt.scatter(t1, trace_mz)

        plt.figure('input')
        plt.title('input signals')
        plt.plot(s_in)
        plt.ylabel('inputs')
        plt.xlabel('Time')
        plt.show()


def multi_system_oscillator_test(length_int=10, node_size=20, task='Delay', superposition=0, ac_amplitude=0.0,
                                 visual_process=True):
    # hyper parameters
    file_path = 'weight_matrix_multi_device'
    positive_dc_current_oscillator = 200
    negative_dc_current_oscillator = 100
    time_consume_oscillator = 4e-9
    f_ac = 32e9

    # build two reservoir physical devices
    switch_device = o_mtj.Mtj()
    oscillator_device = o_mtj.Mtj()

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if os.path.exists(f'{file_path}/Oscillator_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy'):
        weight_out_stm = np.load(
            f'{file_path}/Oscillator_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy')
        print('###############################################')
        print(f'{file_path}/Oscillator_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy loading !')
        print('###############################################')

    else:
        sys.exit('no valid data')

    # it seems to build a function better

    s_in, train_signal = o_mtj.real_time_generator(task=f'{task}', superposition_number=superposition,
                                                   length_signals=length_int)
    # pre training
    for pre_length in range(5):
        dc_current_list = [positive_dc_current_oscillator, negative_dc_current_oscillator]
        dc_current_value = dc_current_list[np.random.randint(0, 2, 1)[0]]
        oscillator_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                         time_consumed=time_consume_oscillator, f_ac=f_ac)
        switch_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                     time_consumed=time_consume_oscillator)

    trace_mz, y_out_list = [], []

    for i in track(range(len(s_in))):
        if s_in[i] == 1:
            dc_current_value = positive_dc_current_oscillator
            _, _, mz_list, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_oscillator,
                                                                       f_ac=f_ac)
            _, _, mx_list, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_oscillator)

        else:
            dc_current_value = negative_dc_current_oscillator
            _, _, mz_list, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_oscillator,
                                                                       f_ac=f_ac)
            _, _, mx_list, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_oscillator)

        # for oscillator
        try:
            mz_list_all, t_list_whole = [], []
            if 'extreme_high' not in locals().keys():
                extreme_high, extreme_low = [], []
            else:
                extreme_high, extreme_low = [extreme_high[-1]], [extreme_low[-1]]

            mz_list, t_list_o = list(mz_list), list(t_list_o)

            for i1 in range(len(mz_list)):
                if i1 != 0 and i1 != len(mz_list) - 1:
                    if mz_list[i1] > mz_list[i1 - 1] and mz_list[i1] > mz_list[i1 + 1]:
                        extreme_high.append(mz_list[i1])
                    if mz_list[i1] < mz_list[i1 - 1] and mz_list[i1] < mz_list[i1 + 1]:
                        extreme_low.append(mz_list[i1])

            length_extreme = min(len(extreme_low), len(extreme_high))
            for i2 in range(length_extreme):
                mz_list_all.append(extreme_high[i2])

            xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
            fp = mz_list_all
            sampling_x_values = np.linspace(1, len(mz_list_all), node_size)
            f = interp1d(xp, fp, kind='quadratic')
            x_matrix_o = f(sampling_x_values)
            trace_mz = trace_mz + list(x_matrix_o)

        except Exception as ErrorMessage:
            print('error in building oscillator reservoir')
            sys.exit(ErrorMessage)

        # for switch device
        try:
            mz_list_all, t_list_whole = [], []
            if 'extreme_high' not in locals().keys():
                extreme_high, extreme_low = [], []
            else:
                extreme_high, extreme_low = [extreme_high[-1]], [extreme_low[-1]]

            mx_list, t_list_o = list(mx_list), list(t_list_o)

            for i1 in range(len(mx_list)):
                if i1 != 0 and i1 != len(mx_list) - 1:
                    if mx_list[i1] > mx_list[i1 - 1] and mx_list[i1] > mx_list[i1 + 1]:
                        extreme_high.append(mx_list[i1])
                    if mx_list[i1] < mx_list[i1 - 1] and mx_list[i1] < mx_list[i1 + 1]:
                        extreme_low.append(mx_list[i1])

            length_extreme = min(len(extreme_low), len(extreme_high))
            for i2 in range(length_extreme):
                mz_list_all.append(extreme_high[i2])

            xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
            fp = mz_list_all
            sampling_x_values = np.linspace(1, len(mz_list_all), node_size)
            f = interp1d(xp, fp, kind='quadratic')
            x_matrix_s = f(sampling_x_values)
            trace_mz = trace_mz + list(x_matrix_s)

        except Exception as ErrorMessage:
            print('error in building switch_device reservoir')
            sys.exit(ErrorMessage)

        # calculate output and update weights
        try:
            x_matrix = np.vstack([x_matrix_s, x_matrix_o])
            x_matrix = np.append(x_matrix.T, 1).reshape(-1, 1)
            y_out = np.dot(weight_out_stm, x_matrix)
            y_out_list.append(y_out[0, 0])

        except ValueError as ErrorMessage:
            print('----------------------------------------')
            print('error from readout layer: {}'.format(ErrorMessage))
            print('Please check for your weight file at {}'.format(file_path))
            print('________________________________________')
            sys.exit(ErrorMessage)

    # calculate the error -> mean square error
    error_learning = np.var(np.array(train_signal) - np.array(y_out_list))
    print('##################################################################')
    print('##################################################################')
    capacity = pow(np.corrcoef(y_out_list[superposition:], train_signal[superposition:])[0, 1], 2)

    # visualization of magnetization
    if visual_process:
        plt.figure('Test results')
        plt.plot(train_signal[superposition:], c='blue', label='target')
        plt.plot(y_out_list[superposition:], c='green', label='module')
        plt.ylabel('signals')
        plt.xlabel('Time')
        plt.legend()

        plt.figure('Comparison')
        plt.subplot(2, 1, 1)
        plt.title('Mean square error : {}'.format(error_learning))
        plt.plot(train_signal[superposition:], c='blue', label='target output')
        plt.legend()
        plt.ylabel('signals')
        plt.xlabel('Time')

        plt.subplot(2, 1, 2)
        plt.plot(y_out_list[superposition:], c='red', label='actual output')
        # plt.title('Mean square error : {}'.format(error_learning))
        plt.legend()
        plt.ylabel('signals')
        plt.xlabel('Time')
        plt.show()

    return capacity


def multi_system_switch(length_int=100, node_size=20, task='Delay', superposition=0, recover_weight=False,
                        ac_amplitude=0.0,
                        save_index=True, visual_process=True):
    # hyper parameters
    file_path = 'weight_matrix_multi_device'
    positive_dc_current_switch, negative_dc_current_switch = 200, -200
    time_consume_switch = 8e-9
    f_ac = 32e9

    # build two reservoir physical devices
    switch_device = s_mtj.Mtj()
    oscillator_device = s_mtj.Mtj()

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if os.path.exists(f'{file_path}/Switch_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy'):
        weight_out_stm = np.load(f'{file_path}/Switch_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy')
        print('###############################################')
        print(f'{file_path}/Switch_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy already exists !')
        print('###############################################')

        if not recover_weight:
            return 0
        else:
            print('NO MATTER to retrain again !')

    else:
        # think about bias term
        weight_out_stm = np.random.randint(-1, 2, (1, node_size * 2 + 1))
        print('\r weight matrix of STM created successfully !', end='', flush=True)

    # it seems to build a function better

    s_in, train_signal = o_mtj.real_time_generator(task=f'{task}', superposition_number=superposition,
                                                   length_signals=length_int)
    # pre training
    for pre_length in range(5):
        dc_current_list = [positive_dc_current_switch, negative_dc_current_switch]
        dc_current_value = dc_current_list[np.random.randint(0, 2, 1)[0]]
        switch_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                     time_consumed=time_consume_switch)
        dc_current_value = dc_current_list[np.random.randint(0, 2, 1)[0]]
        oscillator_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                         time_consumed=time_consume_switch, f_ac=f_ac)

    trace_mz, y_out_list, x_update_matrix = [], [], []

    for i in track(range(len(s_in))):
        if s_in[i] == 1:
            dc_current_value = positive_dc_current_switch
            mz_list, _, _, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_switch,
                                                                       f_ac=f_ac)

            mx_list, _, _, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_switch)

        else:
            dc_current_value = negative_dc_current_switch
            mz_list, _, _, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_switch,
                                                                       f_ac=f_ac)

            mx_list, _, _, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_switch)

        # for oscillator
        try:
            index_sample = np.linspace(0, len(mz_list) - 1, node_size, dtype=int, endpoint=False)
            x_matrix_o = np.array([mz_list[index_value] for index_value in index_sample])
            trace_mz = trace_mz + list(x_matrix_o)

        except Exception as ErrorMessage:
            print('error in building oscillator reservoir')
            sys.exit(ErrorMessage)

        # for switch device
        try:
            index_sample = np.linspace(0, len(mx_list) - 1, node_size, dtype=int, endpoint=False)
            x_matrix_s = np.array([mx_list[index_value] for index_value in index_sample])
            trace_mz = trace_mz + list(x_matrix_s)

        except Exception as ErrorMessage:
            print('error in building switch_device reservoir')
            sys.exit(ErrorMessage)

        # calculate output and update weights
        try:
            x_matrix = np.vstack([x_matrix_s, x_matrix_o])
            x_matrix = np.append(x_matrix.T, 1).reshape(-1, 1)
            y_out = np.dot(weight_out_stm, x_matrix)
            y_out_list.append(y_out[0, 0])
            x_update_matrix.append(x_matrix.T.tolist()[0])

        except ValueError as error:
            print('----------------------------------------')
            print('error from readout layer: {}'.format(error))
            print('Please check for your weight file at {}'.format(file_path))
            print('________________________________________')
            return 0

    # update weight
    y_train_matrix = np.array(train_signal).reshape(1, len(train_signal))
    x_update_matrix = np.asmatrix(x_update_matrix).T
    weight_out_stm = np.dot(y_train_matrix, np.linalg.pinv(x_update_matrix))

    y_test = np.dot(weight_out_stm, x_update_matrix)
    print('train result:{}'.format(y_test))

    # calculate the error -> mean square error
    error_learning = np.var(np.array(train_signal) - np.array(y_test))
    print('##################################################################')
    print('error:{}'.format(error_learning))
    print('Trained successfully !')
    print('##################################################################')

    # save weight matrix as .npy files
    if save_index:
        np.save(f'{file_path}/Switch_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy', weight_out_stm)
        print('Saved weight matrix file')

    # visualization of magnetization
    if visual_process:
        plt.figure('Trace of mz')
        plt.title('reservoirs states')
        plt.xlabel('Time interval')
        plt.ylabel(r'$m_z$')
        t1 = np.linspace(0, len(trace_mz) - 1, len(trace_mz))
        plt.scatter(t1, trace_mz)

        plt.figure('input')
        plt.title('input signals')
        plt.plot(s_in)
        plt.ylabel('inputs')
        plt.xlabel('Time')
        plt.show()


def multi_system_switch_test(length_int=10, node_size=20, task='Delay', superposition=0, ac_amplitude=0.0,
                             visual_process=True):
    # hyper parameters
    file_path = 'weight_matrix_multi_device'
    positive_dc_current_switch, negative_dc_current_switch = 200, -200
    time_consume_switch = 8e-9
    f_ac = 32e9

    # build two reservoir physical devices
    switch_device = s_mtj.Mtj()
    oscillator_device = s_mtj.Mtj()

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if os.path.exists(f'{file_path}/Switch_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy'):
        weight_out_stm = np.load(f'{file_path}/Switch_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy')
        print('###############################################')
        print(f'{file_path}/Switch_{task}_{superposition}_node_{node_size}_ac_{ac_amplitude}.npy loading !')
        print('###############################################')

    else:
        sys.exit('no valid data')

    # it seems to build a function better

    s_in, train_signal = o_mtj.real_time_generator(task=f'{task}', superposition_number=superposition,
                                                   length_signals=length_int)
    # pre training
    for pre_length in range(5):
        dc_current_list = [positive_dc_current_switch, negative_dc_current_switch]
        dc_current_value = dc_current_list[np.random.randint(0, 2, 1)[0]]
        switch_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                     time_consumed=time_consume_switch)
        oscillator_device.time_evolution(dc_amplitude=dc_current_value, ac_amplitude=ac_amplitude,
                                         time_consumed=time_consume_switch, f_ac=f_ac)

    trace_mz, y_out_list, x_update_matrix = [], [], []

    for i in track(range(len(s_in))):
        if s_in[i] == 1:
            dc_current_value = positive_dc_current_switch
            mz_list, _, _, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_switch,
                                                                       f_ac=f_ac)

            mx_list, _, _, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_switch)

        else:
            dc_current_value = negative_dc_current_switch
            mz_list, _, _, t_list_o = oscillator_device.time_evolution(dc_amplitude=dc_current_value,
                                                                       ac_amplitude=ac_amplitude,
                                                                       time_consumed=time_consume_switch,
                                                                       f_ac=f_ac)

            mx_list, _, _, t_list_s = switch_device.time_evolution(dc_amplitude=dc_current_value,
                                                                   ac_amplitude=ac_amplitude,
                                                                   time_consumed=time_consume_switch)

        # for oscillator
        try:
            index_sample = np.linspace(0, len(mz_list) - 1, node_size, dtype=int, endpoint=False)
            x_matrix_o = np.array([mz_list[index_value] for index_value in index_sample])
            trace_mz = trace_mz + list(x_matrix_o)

        except Exception as ErrorMessage:
            print('error in building oscillator reservoir')
            sys.exit(ErrorMessage)

        # for switch device
        try:
            index_sample = np.linspace(0, len(mx_list) - 1, node_size, dtype=int, endpoint=False)
            x_matrix_s = np.array([mx_list[index_value] for index_value in index_sample])
            trace_mz = trace_mz + list(x_matrix_s)

        except Exception as ErrorMessage:
            print('error in building switch_device reservoir')
            sys.exit(ErrorMessage)

        # calculate output and update weights
        try:
            x_matrix = np.vstack([x_matrix_s, x_matrix_o])
            x_matrix = np.append(x_matrix.T, 1).reshape(-1, 1)
            y_out = np.dot(weight_out_stm, x_matrix)
            y_out_list.append(y_out[0, 0])

        except ValueError as ErrorMessage:
            print('----------------------------------------')
            print('error from readout layer: {}'.format(ErrorMessage))
            print('Please check for your weight file at {}'.format(file_path))
            print('________________________________________')
            sys.exit(ErrorMessage)

    # calculate the error -> mean square error
    error_learning = np.var(np.array(train_signal) - np.array(y_out_list))
    print('##################################################################')
    print('error:{}'.format(error_learning))
    print('##################################################################')
    capacity = pow(np.corrcoef(y_out_list[superposition:], train_signal[superposition:])[0, 1], 2)

    # visualization of magnetization
    if visual_process:
        plt.figure('Test results')
        plt.plot(train_signal[superposition:], c='blue', label='target')
        plt.plot(y_out_list[superposition:], c='green', label='module')
        plt.ylabel('signals')
        plt.xlabel('Time')
        plt.legend()

        plt.figure('Comparison')
        plt.subplot(2, 1, 1)
        plt.title('Mean square error : {}'.format(error_learning))
        plt.plot(train_signal[superposition:], c='blue', label='target output')
        plt.legend()
        plt.ylabel('signals')
        plt.xlabel('Time')

        plt.subplot(2, 1, 2)
        plt.plot(y_out_list[superposition:], c='red', label='actual output')
        # plt.title('Mean square error : {}'.format(error_learning))
        plt.legend()
        plt.ylabel('signals')
        plt.xlabel('Time')
        plt.show()

    return capacity


if __name__ == '__main__':
    # draw a picture
    mtj_demo = s_mtj.Mtj(x0=-1, y0=0.01, z0=0.01)
    mx_list_whole = []
    random_inputs = [1]
    # random_inputs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    print(random_inputs)
    for i in random_inputs:
        if i == 1:
            dc_value = -200
        else:
            dc_value = 200
        mx_set, _, mz_list_set, _ = mtj_demo.time_evolution(dc_amplitude=dc_value, time_consumed=8e-9)
        mx_list_whole = np.append(mx_list_whole, mx_set)

    mtj_demo = o_mtj.Mtj()
    _, _, mz_list_set, _ = mtj_demo.time_evolution(dc_amplitude=200, time_consumed=2e-8)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.ylim(-1.1, 1.1)
    t1 = np.linspace(0, 1, len(mx_list_whole))
    plt.plot(t1, mx_list_whole, c='green')
    plt.title('Switching')
    plt.xlabel('Time')
    plt.ylabel('Magnetization')
    plt.subplot(1, 2, 2)
    t1 = np.linspace(0, 1, len(mz_list_set))
    plt.plot(t1, mz_list_set, c='orange')
    plt.ylim(-1.1, 1.1)
    plt.title('Oscillating')
    plt.xlabel('Time')
    plt.show()

    work_path = 'D:\\Python_projects\\Physical_RC_data'
    if os.path.exists(work_path):
        os.chdir(work_path)
    print(f'Path: {os.getcwd()}')

    multi_system_switch(length_int=500, node_size=2, recover_weight=False, visual_process=True,
                        superposition=1,
                        task='Parity')
    covariance = multi_system_oscillator_test(length_int=60, node_size=2, superposition=0,
                                              visual_process=True,
                                              task='Parity')
    print(covariance)

    node_list = [2, 5, 10, 16, 30, 50]
    superposition_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    task_list = ['Delay', 'Parity']
    for number_value in range(10):
        for node in node_list:
            for task in task_list:
                covariance_list = []
                for superposition_value in superposition_list:
                    # multi_system_switch(length_int=500, node_size=node, recover_weight=False, visual_process=False,
                    #                     superposition=superposition_value,
                    #                     task=task)
                    # covariance = multi_system_switch_test(length_int=60, node_size=node,
                    #                                       superposition=superposition_value,
                    #                                       visual_process=False,
                    #                                       task=task)
                    # multi_system_oscillator(length_int=500, node_size=node, recover_weight=False,
                    #                         visual_process=False,
                    #                         superposition=superposition_value,
                    #                         task=task)
                    # covariance = multi_system_oscillator_test(length_int=60, node_size=node,
                    #                                           superposition=superposition_value,
                    #                                           visual_process=False,
                    #                                           task=task)

                    multi_system(length_int=500, node_size=node, recover_weight=False, visual_process=False,
                                 superposition=superposition_value,
                                 task=task)
                    covariance = multi_system_test(length_int=60, node_size=node,
                                                   superposition=superposition_value,
                                                   visual_process=False,
                                                   task=task)

                    covariance_list.append(covariance)
                    print(covariance)
                df = pd.DataFrame({'number': superposition_list, 'covariance': covariance_list})
                df.to_excel(f'Multi_node_{node}_{task}_{number_value}.xlsx')
