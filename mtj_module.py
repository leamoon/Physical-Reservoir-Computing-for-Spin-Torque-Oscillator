import os
import sys
import numpy as np
import pandas as pd
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.header import Header
from rich.progress import track
from scipy.interpolate import interp1d
from scipy.signal import argrelmax

"""
    This module is built by Xuezhao for simulating the baisc MTJ class.
"""

def email_alert(subject='Default', receiver='1060014562@qq.com'):
    """
    a function sends an alert message to my email.
    :param subject: content of email
    :param receiver: email of receiver
    """
    mail_host = "smtp.qq.com"
    mail_user = "1060014562"
    mail_pass = "dwwsklhrgaoybbjh"
    sender = '1060014562@qq.com'
    message = MIMEText('Training successfully!', 'plain', 'utf-8')
    message['From'] = Header("Robot_Training", 'utf-8')
    message['To'] = Header("Dear Master", 'utf-8')
    message['Subject'] = Header(subject, 'utf-8')

    smtp = SMTP_SSL(mail_host)
    smtp.ehlo(mail_host)
    smtp.login(mail_user, mail_pass)
    print('sending !')

    smtp.sendmail(sender, receiver, message.as_string())
    return 0


def real_time_generator(task='Delay', superposition_number=1, length_signals=100, posibility_0=0.5):
    """
    a function used to associate training delay task and Parity check task
    :param task: 'Delay' or 'Parity' corresponds to different task
    :param superposition_number: delay time for delay task, superposition number for parity check task
    :param length_signals: length of signals
    :return: input signals and train signals
    """
    try:
        # s_in = np.random.randint(0, 2, length_signals)
        s_in = np.random.choice([0, 1], size=length_signals, p=[posibility_0, 1-posibility_0])

        if superposition_number == 0:
            train_signal = s_in
        else:
            if task == 'Delay':
                train_signal = np.append(s_in[-int(superposition_number):], s_in[:-int(superposition_number)])

            elif task == 'Parity':
                train_signal = s_in
                for super_value in range(1, superposition_number + 1):
                    temp_signal = np.append(s_in[-int(super_value):], s_in[:-int(super_value)])
                    train_signal = train_signal + temp_signal
                    train_signal[np.argwhere(train_signal == 2)] = 0

        return s_in, train_signal

    except Exception as error:
        print('Sth wrong in generating input signals:{}'.format(error))
        sys.exit(0)


class Mtj:
    def __init__(self, x0=0.1, y0=0.1, z0=1, t_step=3e-13):
        # initial states of magnetization
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0 + 2 # for positive PMA
        self.t_step = t_step

        # hyper parameters
        self.damping_factor = 0.02
        self.gyo_ratio = 1.7e7  # unit Oe^-1 s^-1

        # declare variables
        self.external_field, self.anisotropy_field, self.demagnetization_field = 0, 0, 0
        self.effective_field = 0
        self.time_ac = 0
        self.stt_amplitude, self.ac_amplitude, self.dc_amplitude = 0, 0, 0
        self.dc_frequency, self.ac_frequency = 0, 0

        self.theta, self.phi = None, None

        # normalization of magnetization vector
        self.m = np.array([self.x0, self.y0, self.z0])
        self.m = self.m / np.linalg.norm(self.m)
        self.x0, self.y0, self.z0 = self.m

    def parameters_calculation(self, external_field=200, anisotropy_field=0, demagnetization_field=8400,
                               dc_amplitude=100, time_ac=0, ac_amplitude=0, ac_frequency=32e9):
        # effective field
        self.external_field = external_field
        self.anisotropy_field = anisotropy_field
        self.demagnetization_field = demagnetization_field
        self.effective_field = np.array([self.external_field + self.anisotropy_field * self.x0, 0,
                                         -self.demagnetization_field * self.z0])

        # STT term
        self.dc_amplitude = dc_amplitude
        self.ac_amplitude = ac_amplitude
        self.time_ac = time_ac
        self.ac_frequency = ac_frequency
        self.stt_amplitude = dc_amplitude + ac_amplitude * np.cos(time_ac * ac_frequency)

    def step_evolution(self, time_ac, magnetization, dc_amplitude, ac_amplitude, f_ac):
        # time evolution from differential equation
        self.m = magnetization / np.linalg.norm(magnetization)
        self.x0, self.y0, self.z0 = self.m
        self.ac_frequency = f_ac
        self.parameters_calculation(time_ac=time_ac, dc_amplitude=dc_amplitude, ac_amplitude=ac_amplitude)

        x_axis = np.array([1, 0, 0])
        self.stt_amplitude = self.dc_amplitude + self.ac_amplitude * np.cos(self.time_ac * self.ac_frequency)
        self.stt_amplitude = self.stt_amplitude * self.gyo_ratio

        delta_m = -self.gyo_ratio * np.cross(self.m, self.effective_field) + self.damping_factor * (
                -self.gyo_ratio * np.dot(self.m,
                                         self.effective_field) * self.m + self.gyo_ratio * self.effective_field +
                self.stt_amplitude * np.cross(self.m, (self.m * self.x0 - x_axis))) + self.stt_amplitude * np.cross(
            self.m, np.cross(self.m, x_axis))

        delta1_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))
        return delta1_m_reduce

    def time_evolution(self, dc_amplitude=420.21, ac_amplitude=0.0, time_consumed=1e-8, f_ac=32e9):
        sol = scipy.integrate.solve_ivp(self.step_evolution, t_span=(0, time_consumed), y0=self.m,
                                        t_eval=np.linspace(0, time_consumed, int(time_consumed / self.t_step)),
                                        args=[dc_amplitude, ac_amplitude, f_ac], dense_output=True, atol=1e-10,
                                        rtol=1e-6)
        t_list = sol.t
        mx_list, my_list, mz_list = sol.y
        # normalization
        norms = np.linalg.norm(sol.y, axis=0)
        mx_list, my_list, mz_list = mx_list / norms, my_list / norms, mz_list / norms
        self.x0, self.y0, self.z0 = mx_list[-1], my_list[-1], mz_list[-1]
        self.m = np.array([self.x0, self.y0, self.z0])
        return mx_list, my_list, mz_list, t_list, self.m

    def get_reservoirs(self, dc_current=100, ac_current=0.0, consuming_time=1e-8, size=16, f_ac=32e9):
        mx_list, my_list, mz_list, _, _ = self.time_evolution(dc_amplitude=dc_current, ac_amplitude=ac_current,
                                                              time_consumed=consuming_time, f_ac=f_ac)
        try:
            mz_list_all = mz_list[argrelmax(mz_list)]
            xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
            fp = mz_list_all
            sampling_x_values = np.linspace(1, len(mz_list_all), size)
            # linear slinear quadratic cubic
            f = interp1d(xp, fp, kind='quadratic')
            mz_sampling = f(sampling_x_values)

            reservoir_states = np.zeros((len(mz_sampling), 3))
            reservoir_states[:, -1] = mz_sampling
            reservoir_states[-1, :] = np.array([mx_list[-1], my_list[-1], mz_list[-1]])

            return reservoir_states

        except Exception as error:
            print('----------------error---------------------')
            print('error from sampling: {}'.format(error))
            print('________________error______________________')
            sys.exit()

    def real_time_train(self, number_wave, nodes_stm, file_path='weight_matrix_oscillator_xuezhao',
                        visual_process=False,
                        save_index=True, alert_index=False, superposition=0, time_consume_all=1e-8, ac_amplitude=0.0,
                        f_ac=32e9, recover_weight=False, task='Delay', positive_dc_current=200, negative_dc_current=100, posibility_0=0.5):
        """
        a function used to train weight matrix of readout layer in classification task
        :param file_path: the path of weight_matrix
        :param number_wave: number of wave forms
        :param nodes_stm: Size of reservoirs
        :param visual_process: showing the process figures if it is True, default as False
        :param save_index: save weight matrix file or not
        :param alert_index: sending notification or not after training successfully
        :param superposition: the number of delay or time interval
        :param time_consume_all: time consume in single step evolution
        :param ac_amplitude: amplitude of ac term
        :param f_ac: frequency of ac stt term
        :param recover_weight: boolean, recover all weights if True
        :param task: delay task or parity task
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if os.path.exists(f'{file_path}/STM_{task}_{superposition}_node_{nodes_stm}_ac_{ac_amplitude}.npy'):
            weight_out_stm = np.load(f'{file_path}/STM_{task}_{superposition}_node_{nodes_stm}_ac_{ac_amplitude}.npy')
            print('###############################################')
            print(f'{file_path}/STM_{task}_{superposition}_node_{nodes_stm}_ac_{ac_amplitude}.npy already exists !')
            print('###############################################')

            if not recover_weight:
                return 0
            else:
                print('NO MATTER to retrain again !')

        else:
            # think about bias term
            weight_out_stm = np.random.randint(-1, 2, (1, nodes_stm + 1))
            # print('\r weight matrix of STM created successfully !', end='', flush=True)

        # fabricate the input, target, and output signals
        y_out_list, x_final_matrix, plus_visual_mz, minus_visual_mz = [], [], [], []
        plus_time, minus_time, time_index = [], [], 0

        # # it seems to build a function better
        # positive_dc_current = 200
        # negative_dc_current = 100

        s_in, train_signal = real_time_generator(task=f'{task}', superposition_number=superposition,
                                                 length_signals=number_wave, posibility_0=posibility_0)
        # pre-training
        self.time_evolution(dc_amplitude=positive_dc_current, ac_amplitude=ac_amplitude,
                            time_consumed=1e-8,
                            f_ac=f_ac)

        trace_mz = []

        # for i in track(range(len(s_in))):
        for i in range(len(s_in)):
            if s_in[i] == 1:
                dc_current1 = positive_dc_current
                _, _, mz_list_chao, t_list2, _ = self.time_evolution(dc_amplitude=dc_current1, ac_amplitude=ac_amplitude,
                                                                  time_consumed=time_consume_all, f_ac=f_ac)

            else:
                dc_current1 = negative_dc_current
                _, _, mz_list_chao, t_list2, _ = self.time_evolution(dc_amplitude=dc_current1, ac_amplitude=ac_amplitude,
                                                                  time_consumed=time_consume_all, f_ac=f_ac)

            try:
                mz_list_all, t_list_whole = [], []
                if 'extreme_high' not in locals().keys():
                    extreme_high, extreme_low = [], []
                else:
                    extreme_high, extreme_low = [extreme_high[-1]], [extreme_low[-1]]

                mz_list_chao, t_list2 = list(mz_list_chao), list(t_list2)

                for i1 in range(len(mz_list_chao)):
                    if i1 != 0 and i1 != len(mz_list_chao) - 1:
                        if mz_list_chao[i1] > mz_list_chao[i1 - 1] and mz_list_chao[i1] > mz_list_chao[i1 + 1]:
                            extreme_high.append(mz_list_chao[i1])
                        if mz_list_chao[i1] < mz_list_chao[i1 - 1] and mz_list_chao[i1] < mz_list_chao[i1 + 1]:
                            extreme_low.append(mz_list_chao[i1])

                length_extreme = min(len(extreme_low), len(extreme_high))
                for i2 in range(length_extreme):
                    mz_list_all.append(extreme_high[i2])

            except Exception as error:
                print('error from sampling curve :{}'.format(error))
                mz_list_all = mz_list_chao
                sys.exit(error)

            # trace_mz = trace_mz + mz_list_all

            # for figures
            if s_in[i] == 1:
                plus_visual_mz = plus_visual_mz + mz_list_chao
                plus_time = plus_time + list(
                    np.linspace(time_index, time_index + 3e-13 * (len(t_list2) - 1), len(t_list2)))
            else:
                minus_visual_mz = minus_visual_mz + mz_list_chao
                minus_time = minus_time + list(
                    np.linspace(time_index, time_index + 3e-13 * (len(t_list2) - 1), len(t_list2)))

            time_index = time_index + t_list2[-1]

            # sampling points
            try:
                xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
                fp = mz_list_all
                sampling_x_values = np.linspace(1, len(mz_list_all), nodes_stm)
                # x_matrix1 = np.interp(sampling_x_values, xp, fp)
                # linear slinear quadratic cubic
                f = interp1d(xp, fp, kind='quadratic')
                x_matrix1 = f(sampling_x_values)

                trace_mz = trace_mz + list(x_matrix1)
                # print(f'before: {mz_list_all}')
                # print(f'after: {x_matrix1}')

            except Exception as error:
                print('---------------------------------------')
                print('error from sampling: {}'.format(error))
                print('_______________________________________')
                return 0

            # add bias term
            try:
                x_matrix1 = np.append(x_matrix1.T, 1).reshape(-1, 1)
                y_out = np.dot(weight_out_stm, x_matrix1)
                y_out_list.append(y_out[0, 0])
                x_final_matrix.append(x_matrix1.T.tolist()[0])

            except ValueError as error:
                print('----------------------------------------')
                print('error from readout layer: {}'.format(error))
                print('Please check for your weight file at {}'.format(file_path))
                print('________________________________________')
                return 0

        # update weight
        y_train_matrix = np.array(train_signal).reshape(1, len(train_signal))
        x_final_matrix = np.asmatrix(x_final_matrix).T

        # here is for regression
        # temp_1 = np.dot(y_train_matrix, x_final_matrix.T)
        # weight_out_stm = np.dot(temp_1, np.linalg.pinv(np.dot(x_final_matrix, x_final_matrix.T)))
        weight_out_stm = np.dot(y_train_matrix, np.linalg.pinv(x_final_matrix))

        y_test = np.dot(weight_out_stm, x_final_matrix)
        # print('train result:{}'.format(y_test))

        # calculate the error
        error_learning = np.var(np.array(train_signal) - np.array(y_test))
        # print('##################################################################')
        # print('error:{}'.format(error_learning))
        # print('Trained successfully !')
        # print('##################################################################')

        # save weight matrix as .npy files
        if save_index:
            np.save(f'{file_path}/STM_{task}_{superposition}_node_{nodes_stm}_ac_{ac_amplitude}.npy', weight_out_stm)
            # print('Saved weight matrix file')

        # visualization of magnetization
        if visual_process:
            plt.figure('Trace of mz')
            plt.title('reservoirs states')
            plt.xlabel('Time interval')
            plt.ylabel(r'$m_z$')
            t1 = np.linspace(0, len(trace_mz) - 1, len(trace_mz))
            plt.scatter(t1, trace_mz)
            # plt.show()

            plt.figure('visual')
            plt.title('Magnetization Behavior')
            plt.xlabel('Time')
            plt.ylabel(r'$M_z$')
            plt.scatter(minus_time, minus_visual_mz, label='1', s=2, c='red')
            plt.scatter(plus_time, plus_visual_mz, label='0', s=2, c='blue')
            plt.legend()

            plt.figure('input')
            plt.title('input signals')
            plt.plot(s_in)
            plt.ylabel('inputs')
            plt.xlabel('Time')
            plt.show()

        # notification
        if alert_index:
            email_alert(subject='Training Successfully !')

    def real_time_test(self, test_number=80, nodes_stm=80, file_path='weight_matrix_oscillator_xuezhao',
                       visual_index=True,
                       alert_index=False, superposition=0, time_consume_all=1e-8, ac_amplitude=0.0, f_ac=32e9,
                       task='Delay', positive_dc_current=200, negative_dc_current=100, posibility_0=0.5):
        """
        a function used to test the ability of classification of chaotic-MTJ echo state network
        :param test_number: the number of test waves form, default:80
        :param nodes_stm: Size of classification, default:80
        :param file_path: Path of weight matrix file
        :param visual_index: show test result as figures if it is True
        :param alert_index: sending notification when it is True
        :param superposition: number of time interval or delay time
        :param time_consume_all: time consume in single step evolution
        :param ac_amplitude: amplitude of ac term
        :param f_ac: frequency of ac stt term
        :param task: delay task or parity check task
        :return mean square error between train signals and output signals
        """

        if os.path.exists(f'{file_path}/STM_{task}_{superposition}_node_{nodes_stm}_ac_{ac_amplitude}.npy'):
            weight_out_stm = np.load(f'{file_path}/STM_{task}_{superposition}_node_{nodes_stm}_ac_{ac_amplitude}.npy')
            print('Loading STM_{}_{}_node_{}_ac_{} matrix successfully !'.format(task, superposition, nodes_stm,
                                                                                 ac_amplitude))
            print('shape of weight:{}'.format(weight_out_stm.shape))
            print('###############################################')

        else:
            print('\rno valid weight data !', end='', flush=True)
            sys.exit(-1)

        # fabricate the input, target, and output signals
        y_out_list, x_final_matrix, plus_visual_mz, minus_visual_mz = [], [], [], []
        plus_time, minus_time, time_index = [], [], 0

        # # it seems to build a function better
        # positive_dc_current = 200
        # negative_dc_current = 100

        s_in, train_signal = real_time_generator(task=f'{task}', superposition_number=superposition,
                                                 length_signals=test_number, posibility_0=posibility_0)

        trace_mz = []
        # pre-training
        self.time_evolution(dc_amplitude=positive_dc_current, ac_amplitude=ac_amplitude,
                            f_ac=f_ac, time_consumed=1e-8)
        # time_consume_all = 1e-8

        # for i_1 in track(range(len(s_in))):
        for i_1 in range(len(s_in)):    
            if s_in[i_1] == 1:
                dc_current1 = positive_dc_current
                _, _, mz_list_chao, t_list2, _ = self.time_evolution(dc_amplitude=dc_current1, ac_amplitude=ac_amplitude,
                                                                  f_ac=f_ac, time_consumed=time_consume_all)

            else:
                dc_current1 = negative_dc_current
                _, _, mz_list_chao, t_list2, _ = self.time_evolution(dc_amplitude=dc_current1, ac_amplitude=ac_amplitude,
                                                                  f_ac=f_ac,
                                                                  time_consumed=time_consume_all)

            try:
                mz_list_all, t_list_whole = [], []
                if 'extreme_high' not in locals().keys():
                    extreme_high, extreme_low = [], []
                else:
                    extreme_high, extreme_low = [extreme_high[-1]], [extreme_low[-1]]
                mz_list_chao, t_list2 = list(mz_list_chao), list(t_list2)
                for i1 in range(len(mz_list_chao)):
                    if i1 != 0 and i1 != len(mz_list_chao) - 1:
                        if mz_list_chao[i1] > mz_list_chao[i1 - 1] and mz_list_chao[i1] > mz_list_chao[i1 + 1]:
                            extreme_high.append(mz_list_chao[i1])
                        if mz_list_chao[i1] < mz_list_chao[i1 - 1] and mz_list_chao[i1] < mz_list_chao[i1 + 1]:
                            extreme_low.append(mz_list_chao[i1])

                length_extreme = min(len(extreme_low), len(extreme_high))
                for i2 in range(length_extreme):
                    mz_list_all.append(extreme_high[i2])

            except Exception as error:
                print('error from sampling curve :{}'.format(error))
                mz_list_all = mz_list_chao

            # print('mz_list_all:{}'.format(mz_list_all))
            # print('len of mz sampling points :{}'.format(len(mz_list_all)))
            trace_mz = trace_mz + mz_list_all

            # for figures
            if s_in[i_1] == 1:
                plus_visual_mz = plus_visual_mz + mz_list_chao
                plus_time = plus_time + list(
                    np.linspace(time_index, time_index + 3e-13 * (len(t_list2) - 1), len(t_list2)))
            else:
                minus_visual_mz = minus_visual_mz + mz_list_chao
                minus_time = minus_time + list(
                    np.linspace(time_index, time_index + 3e-13 * (len(t_list2) - 1), len(t_list2)))

            time_index = time_index + t_list2[-1]

            # sampling points
            try:
                xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
                fp = mz_list_all
                sampling_x_values = np.linspace(1, len(mz_list_all), nodes_stm)
                # x_matrix1 = np.interp(sampling_x_values, xp, fp)
                # linear slinear quadratic cubic
                f = interp1d(xp, fp, kind='quadratic')
                x_matrix1 = f(sampling_x_values)

            except Exception as error:
                print('---------------------------------------')
                print('error from sampling: {}'.format(error))
                print('_______________________________________')
                return 0

            # add bias term
            try:
                x_matrix1 = np.append(x_matrix1.T, 1).reshape(-1, 1)
                y_out = np.dot(weight_out_stm, x_matrix1)

                if 0 <= y_out[0, 0] <= 1:
                    y_out_list.append(y_out[0, 0])
                elif y_out[0, 0] > 1:
                    y_out_list.append(1)
                elif y_out[0, 0] < 0:
                    y_out_list.append(0)

                x_final_matrix.append(x_matrix1.T.tolist()[0])

            except ValueError as error:
                print('----------------------------------------')
                print('error from readout layer: {}'.format(error))
                print('Please check for your weight file at {}'.format(file_path))
                print('________________________________________')
                return 0

        # calculate the error
        # error_learning = np.var(np.array(train_signal) - np.array(y_out_list))
        error_learning = (np.square(np.array(train_signal) - np.array(y_out_list))).mean()
        capacity = pow(np.corrcoef(y_out_list, train_signal)[0, 1], 2)
        print('Test Error:{}'.format(error_learning))
        print(f'$Cor^{2}$: {capacity}')
        print('----------------------------------------------------------------')

        if alert_index:
            email_alert(subject='error of STM : {}'.format(error_learning))

        # FIGURES
        if visual_index:
            # file_figure_path = './Figs'
            # if not os.path.exists(file_figure_path):
            #     os.mkdir(file_figure_path)

            FontSize = 15
            LabelSize = 16

            plt.figure('Test results')
            plt.title('Covariance : {}'.format(capacity))
            plt.plot(train_signal, c='blue', label='target')
            plt.plot(y_out_list, c='green', label='module')
            plt.ylabel(r'input', fontsize=FontSize)
            # plt.xlabel('Time', fontsize=FontSize)
            plt.legend()
            # plt.savefig(f'{file_figure_path}/Test results of ac {ac_amplitude}_{task}{superposition}.png', dpi=1200)

            plt.figure('Comparison')
            plt.subplot(3, 1, 1)
            # plt.text(s='(b)', x=-5, y=1.2, fontdict={'size': LabelSize, 'family': 'Times New Roman'})
            # plt.title('Parity Check task', fontdict={'size': LabelSize, 'family': 'Times New Roman'})
            plt.plot(s_in, c='black', label='input')
            plt.legend(loc='upper right')
            plt.ylabel(r'$S_{in}$', fontsize=FontSize)
            # plt.xlabel('Time')

            plt.subplot(3, 1, 2)
            plt.plot(train_signal, c='black', label='target')
            # plt.title('Mean square error : {}'.format(error_learning))
            plt.legend(loc='upper right')
            plt.ylabel(r'$y_{train}$', fontsize=FontSize)

            plt.subplot(3, 1, 3)
            plt.plot(y_out_list, c='red', label='output')

            plt.legend(loc='upper right')
            plt.xlabel('Time', fontsize=FontSize)
            plt.ylabel(r'$y_{out}$', fontsize=FontSize)
            # plt.savefig(f'{file_figure_path}/Comparison of ac {ac_amplitude}_{task}{superposition}.png', dpi=1200)
            plt.show()

        return capacity
