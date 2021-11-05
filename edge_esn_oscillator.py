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


def email_alert(subject='Default', receiver='wumaomaolemoon@gmail.com'):
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


def gram_schmidt(un_o_matrix):
    """
    a Gram-Schmidt Function
    :param un_o_matrix: a un-orthogonal matrix
    :return o_matrix: orthogonal matrix from un_o_matrix
    :return o_matrix_normal: normalized matrix of o_matrix
    """
    try:
        x_1, x2, x3, x4 = un_o_matrix[0, :], un_o_matrix[1, :], un_o_matrix[2, :], un_o_matrix[3, :]
        x4_new = x4
        x3_new = x3 - (np.dot(x3, x4_new.T) / np.dot(x4_new, x4_new.T))[0, 0] * x4_new
        x2_new = x2 - (np.dot(x2, x4_new.T) / np.dot(x4_new, x4_new.T))[0, 0] * x4_new - (np.dot(
            x2, x3_new.T) / np.dot(x3_new, x3_new.T))[0, 0] * x3_new
        x1_new = x_1 - (np.dot(x_1, x4_new.T) / np.dot(x4_new, x4_new.T))[0, 0] * x4_new - (np.dot(
            x_1, x3_new.T) / np.dot(x3_new, x3_new.T))[0, 0] * x3_new - (
                         np.dot(x_1, x2_new.T) / np.dot(x2_new, x2_new.T))[0, 0] * x2_new
        o_matrix = np.vstack([x1_new, x2_new, x3_new, x4_new])

        # normalization
        x1_new = x1_new / np.linalg.norm(x1_new)
        x2_new = x2_new / np.linalg.norm(x2_new)
        x3_new = x3_new / np.linalg.norm(x3_new)
        x4_new = x4_new / np.linalg.norm(x4_new)
        o_matrix_normal = np.vstack([x1_new, x2_new, x3_new, x4_new])

        return o_matrix, o_matrix_normal

    except Exception as e1:
        print('error from Gram-Schmidt {}'.format(e1))
        return 0


def waveform_generator(number_wave):
    """
    a function used to create random wave(sine or square), 1 input corresponds to 8 output
    :param number_wave: number of waves
    :return: wave_points
    """
    # 8 points to express waveform
    random_pulse = np.random.randint(0, 2, int(number_wave))
    print('random:{}'.format(random_pulse))
    wave_points = []
    for pulse in random_pulse:
        if pulse == 0:
            # sine function
            sine_points = [120, 150, 180, 200, 180, 150, 120, 100]
            wave_points = wave_points + sine_points
        elif pulse == 1:
            # square function
            square_points = [200] * 4 + [100] * 4
            wave_points = wave_points + square_points

    print('wave:{}'.format(wave_points))

    # FIGURE
    plt.figure('input-signals')
    wave_signals = []
    for j1 in wave_points:
        temp2 = [j1] * 20
        wave_signals = wave_signals + temp2
    temp1 = np.linspace(0, len(wave_signals) - 1, len(wave_signals))
    plt.title(r'input waveform')
    plt.plot(temp1, wave_signals)
    plt.xlabel(r'Time')
    plt.ylabel(r'inputs')
    # plt.scatter(temp1, wave_signals)

    return wave_points, list(random_pulse)


def real_time_generator(task='Delay', superposition_number=1, length_signals=100):
    """
    a function used to associate training delay task and Parity check task
    :param task: 'Delay' or 'Parity' corresponds to different task
    :param superposition_number: delay time for delay task, superposition number for parity check task
    :param length_signals: length of signals
    :return: input signals and train signals
    """
    try:
        s_in = np.random.randint(0, 2, length_signals)

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

        # print('############################################################')
        # print('inputs :{}'.format(s_in))
        # print('target :{}'.format(train_signal))
        # print('############################################################')

        return s_in, train_signal

    except Exception as error:
        print('Sth wrong in generating input signals:{}'.format(error))
        sys.exit(0)


def chaos_mine(initial_dif=1e-8, time_consume_single=1e-8, ac_current1=0.0, f_ac=32e9, size=16, input_mode='random',
               save_as_excel=True):
    """
    a function used to find edge of chaos, calculation should start at reservoirs rather than magnetization.
    :param initial_dif: the initial difference / perturbation
    :param time_consume_single:  time used to make evolution of magnetization
    :param ac_current1: amplitude of ac current, unit: Oe
    :param f_ac: frequency of ac stt term
    :param size: size of reservoirs
    :param input_mode: random input or periodic input
    :param save_as_excel: save data in excel file if True
    :return: largest lyapunov exponent
    """

    # hyper parameters
    positive_dc_current = 200
    negative_dc_current = 100
    time_step = 3e-13

    # random initial state to test my algorithm
    initial_state = np.random.random(3)

    # enter into an orbit
    trace = Mtj(initial_state[0], initial_state[1], initial_state[2])
    trace_perturbation = Mtj(initial_state[0], initial_state[1], initial_state[2])
    state = trace.get_reservoirs(dc_current=positive_dc_current, ac_current=ac_current1,
                                 consuming_time=time_consume_single, size=size, f_ac=f_ac)
    state_perturbation = trace_perturbation.get_reservoirs(dc_current=positive_dc_current, ac_current=ac_current1,
                                                           consuming_time=time_consume_single, size=size, f_ac=f_ac)

    # adding perturbation and normalization for reservoir states
    state_perturbation[-1, :] = state_perturbation[-1, :] + np.array([0, 0, initial_dif])
    state_perturbation[-1, :] = state_perturbation[-1, :] / np.linalg.norm(state_perturbation[-1, :])
    real_initial_dif = np.linalg.norm(state_perturbation[-1, :] - state[-1, :])

    # adding perturbation and normalization for magnetization
    trace_perturbation.m = trace_perturbation.m + np.array([0, 0, initial_dif])
    trace_perturbation.m = trace_perturbation.m / np.linalg.norm(trace_perturbation.m)
    real_initial_dif_mag = np.linalg.norm(trace_perturbation.m - trace.m)

    # info
    print('######################--info--################################')
    print('initial state: {}'.format(initial_state))
    print('Time lengthen : {}'.format(int(time_consume_single / time_step)))
    print('trace_state : \n{}'.format(state))
    print('trace_state_perturbation : \n{}'.format(state_perturbation))
    print('Real initial difference : {}'.format(real_initial_dif))
    print('################################################################')

    # loop to calculate le, start from reservoir state
    try:
        iteration_number = int(sys.argv[1])
    except IndexError or ValueError:
        print('default iteration_number=1000')
        iteration_number = 1000

    le_buffer = []
    le_save_excel, index_excel = [], []

    # judge the input mode: periodic or random
    for i1 in range(iteration_number):
        # trace.m = state[-1, :]
        # trace_perturbation.m = state_perturbation[-1, :]

        # random input dc amplitude
        if input_mode == 'random':
            s_in = np.random.randint(0, 2, 1)[0]
            if s_in == 0:
                dc_current_input = positive_dc_current
            elif s_in == 1:
                dc_current_input = negative_dc_current
            else:
                print('some errors in s_in in chaos_mine')

        elif input_mode == 'periodic':
            if i1 % 2 == 0:
                dc_current_input = positive_dc_current
            else:
                dc_current_input = negative_dc_current

        else:
            print('error in input model in lyapunov calculation parts')
            sys.exit()

        state = trace.get_reservoirs(dc_current=dc_current_input, ac_current=ac_current1,
                                     consuming_time=time_consume_single, size=size, f_ac=f_ac)
        state_perturbation = trace_perturbation.get_reservoirs(dc_current=dc_current_input, ac_current=ac_current1,
                                                               consuming_time=time_consume_single, size=size, f_ac=f_ac)
        # here only calculate the last reservoir state
        difference_trace = np.linalg.norm(state_perturbation[-1, :] - state[-1, :])
        if difference_trace == 0:
            print('Error : the initial difference is too small')
            sys.exit()

        le = np.log(difference_trace / real_initial_dif)
        le_buffer.append(le)

        # reset trace_perturbation to avoid numerical overflow
        difference_trace_mag = np.linalg.norm(trace_perturbation.m - trace.m)
        trace_perturbation.m = trace.m + real_initial_dif_mag / difference_trace_mag * (
                trace_perturbation.m - trace.m)
        trace_perturbation.m = trace_perturbation.m / np.linalg.norm(trace_perturbation.m)

        # info
        if i1 % 100 == 1:
            print('##################################### info ########################################')
            print('ac amplitude: {} Oe  f_ac : {} Hz  dc : {} Oe'.format(ac_current1, f_ac, dc_current_input))
            print('input mode: {}'.format(input_mode))
            print('Epoch: {} Current value : {} Average value: {}'.format(i1 + 1, le, np.mean(le_buffer)))

        if i1 == 100:
            le_buffer = []
            print('clear buffer')

        if i1 % 100 == 1 and save_as_excel:
            le_save_excel.append(np.mean(le_buffer))
            index_excel.append(i1)
            df = pd.DataFrame({'Time': index_excel, 'Le': le_save_excel})
            df.to_excel('lyapunov_random_ac{}_{}_f_{}.xlsx'.format(ac_current1, input_mode, f_ac))

        # stop iteration if error less than 0.001
        if i1 % 100 == 1 and len(le_save_excel) >= 2:
            last_le_mean = le_save_excel[-2]
            current_le_mean = le_save_excel[-1]
            if float('{:.2}'.format(last_le_mean)) == float('{:.2}'.format(current_le_mean)):
                print(' lyapunov number is converged')
                break

    return np.mean(le_buffer)


def get_best_reservoir_info(task='Delay', max_time=6, re_train=False):
    """
    a function used to find best reservoir size of echo state network
    :param task: Delay task or Parity task
    :param max_time: the maximum superposition time of memory covariance
    :param re_train: an indication of whether re-training or not
    """
    superposition_time = np.linspace(0, max_time, max_time + 1, dtype=int)
    # superposition_time = [2, 3, 4, 5, 6]
    node_list = [10, 20, 30, 40, 50, 70, 90, 100]
    for superposition_number in superposition_time:
        cov_list = []
        for node in node_list:
            capacity_list = []
            for number in range(1):
                mtj_demo = Mtj()

                mtj_demo.real_time_train(number_wave=500, nodes_stm=node, visual_process=False, save_index=True,
                                         alert_index=False, superposition=superposition_number, time_consume_all=6e-9,
                                         file_path='weight_matrix_interpolation',
                                         ac_amplitude=0.0, task=task, recover_weight=re_train)

                covariance = mtj_demo.real_time_test(test_number=100, nodes_stm=node,
                                                     superposition=superposition_number,
                                                     visual_index=False, file_path='weight_matrix_interpolation',
                                                     ac_amplitude=0.0, time_consume_all=6e-9, task=task)
                capacity_list.append(covariance)
            cov_list.append(np.mean(capacity_list))

        data = {'node': node_list, 'covariance': cov_list}
        df = pd.DataFrame(data)
        df.to_excel(f'Best_size_{task}_{superposition_number}.xlsx')
        print(f'{task}_{superposition_number} save successfully !')


def get_best_time(task='Delay', re_train=False):
    superposition_time = [1, 2, 3, 4]
    node = 16
    file_path = 'weight_matrix_time'
    time_list = [2e-9, 3e-9, 4e-9, 6e-9, 7e-9, 10e-9, 20e-9, 50e-9]
    for time_cal in range(10):
        for time_evolution in time_list:
            file_save_path = os.path.join(file_path, f'{time_evolution}')
            capacity_list = []
            for superposition_number in superposition_time:
                mtj = Mtj()
                mtj.real_time_train(number_wave=500, nodes_stm=node, visual_process=False, save_index=True,
                                    alert_index=False, superposition=superposition_number,
                                    time_consume_all=time_evolution,
                                    file_path=file_save_path,
                                    ac_amplitude=0.0, task=task, recover_weight=re_train)
                capacity = mtj.real_time_test(test_number=100, nodes_stm=node, superposition=superposition_number,
                                              visual_index=False, file_path=file_save_path,
                                              ac_amplitude=0.0, time_consume_all=time_evolution, task=task)
                capacity_list.append(capacity)

            data = pd.DataFrame({'superposition': superposition_time, 'covariance': capacity_list})
            data.to_excel(f'Finding_best_time_{time_cal}_{time_evolution}_{task}.xlsx')


class Mtj:
    def __init__(self, x0=0.1, y0=0.1, z0=1, t_step=3e-13):
        # initial states of magnetization
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
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
        return mx_list, my_list, mz_list, t_list

    def get_reservoirs(self, dc_current=100, ac_current=0.0, consuming_time=1e-8, size=16, f_ac=32e9):
        mx_list, my_list, mz_list, _ = self.time_evolution(dc_amplitude=dc_current, ac_amplitude=ac_current,
                                                           time_consumed=consuming_time, f_ac=f_ac)
        try:
            index_list_high, index_list_low = [], []

            for i1 in range(len(mz_list)):
                if i1 != 0 and i1 != len(mz_list) - 1:
                    if mz_list[i1] > mz_list[i1 - 1] and mz_list[i1] > mz_list[i1 + 1]:
                        index_list_high.append(i1)
                    if mz_list[i1] < mz_list[i1 - 1] and mz_list[i1] < mz_list[i1 + 1]:
                        index_list_low.append(i1)

        except Exception as error:
            print('error from sampling curve :{}'.format(error))
            sys.exit()

        # sampling points
        try:
            if len(index_list_high) < size:
                print('the size of reservoirs is defined too large')
                print('len of m_z : {}'.format(len(index_list_high)))
                print('length of nodes number : {}'.format(size))
                return 0

            number_interval = int(len(index_list_high) / size)
            final_numbers = number_interval * size
            index_points = index_list_high[0: final_numbers: number_interval]

            reservoir_states = np.zeros((len(index_points), 3))
            for i1 in range(len(index_points)):
                state = np.array([mx_list[index_points[i1]], my_list[index_points[i1]], mz_list[index_points[i1]]])
                reservoir_states[i1, :] = state

            return reservoir_states

        except Exception as error:
            print('----------------error---------------------')
            print('error from sampling: {}'.format(error))
            print('________________error______________________')
            sys.exit()

    def lyapunov_parameter(self, current_t, delta_x=None, dc_amplitude=269, ac_amplitude=0):
        """
        a function to get largest lyapunov exponent and its spectrum
        :param current_t: the current time, which work for stt term
        :param delta_x: the input orthogonal matrix
        :param dc_amplitude: amplitude of dc current
        :param ac_amplitude: amplitude of ac current
        :return v_matrix, delta_x
        """

        # define the g parameters, to simplified latter calculation
        temp_magnetization = [self.x0, self.y0, self.z0]  # record the states
        self.x0, self.y0, self.z0 = 0.1, 0.1, 1
        mode = pow(self.x0, 2) + pow(self.y0, 2) + pow(self.z0, 2)
        self.x0 = self.x0 / np.sqrt(mode)
        self.y0 = self.y0 / np.sqrt(mode)
        self.z0 = self.z0 / np.sqrt(mode)
        self.m = np.array([self.x0, self.y0, self.z0])
        self.time_evolution(dc_amplitude=dc_amplitude, ac_amplitude=ac_amplitude, time_consumed=current_t)

        self.stt_amplitude = self.dc_amplitude + self.ac_amplitude * np.cos(self.ac_frequency * current_t)
        stt_partial = -self.ac_amplitude * self.ac_frequency * np.sin(self.ac_frequency * current_t)

        g1 = (-self.gyo_ratio * self.z0 * self.y0 * self.demagnetization_field -
              self.damping_factor * self.gyo_ratio * (self.x0 * self.external_field +
                                                      pow(self.x0, 2) * self.anisotropy_field +
                                                      self.demagnetization_field * pow(self.z0, 2)) * self.x0 +
              self.damping_factor * self.gyo_ratio * (self.external_field + self.anisotropy_field * self.x0) -
              (pow(self.y0, 2) + pow(self.z0, 2)) * self.stt_amplitude * self.damping_factor) / (
                     1 + self.damping_factor ** 2)

        g2 = (-self.gyo_ratio * (-self.x0 * self.z0 * self.demagnetization_field +
                                 self.z0 * self.external_field + self.z0 * self.x0 * self.anisotropy_field) -
              self.damping_factor * self.gyo_ratio * (self.x0 * self.external_field +
                                                      self.anisotropy_field * pow(self.x0, 2) +
                                                      self.demagnetization_field * pow(self.z0, 2)) * self.y0 -
              self.damping_factor * self.gyo_ratio * self.stt_amplitude * self.z0 +
              self.x0 * self.y0 * self.stt_amplitude * self.damping_factor) / (1 + self.damping_factor ** 2)

        g3 = (self.gyo_ratio * (self.y0 * self.external_field + self.x0 * self.y0 * self.anisotropy_field) -
              self.damping_factor * self.gyo_ratio * (self.x0 * self.external_field +
                                                      self.anisotropy_field * pow(self.x0, 2) +
                                                      self.demagnetization_field * pow(self.z0, 2)) * self.z0 +
              self.damping_factor * self.gyo_ratio * (self.demagnetization_field * self.z0 +
                                                      self.stt_amplitude * self.y0) +
              self.stt_amplitude * self.damping_factor * self.x0 * self.z0) / (1 + self.damping_factor ** 2)

        # construct jacobian matrix
        partial_g1_mx = -self.damping_factor * self.gyo_ratio * (
                2 * self.x0 * self.external_field + 3 * pow(self.x0, 2) * self.anisotropy_field +
                self.demagnetization_field * pow(self.z0, 2) - self.anisotropy_field) / (
                                1 + self.damping_factor ** 2)

        partial_g1_my = (-self.gyo_ratio * self.z0 * self.demagnetization_field -
                         2 * self.y0 * self.damping_factor * self.stt_amplitude) / (1 + self.damping_factor ** 2)

        partial_g1_mz = (-self.gyo_ratio * self.y0 * self.demagnetization_field -
                         self.damping_factor * self.gyo_ratio * 2 * self.z0 * self.x0 * self.demagnetization_field -
                         self.z0 * 2 * self.stt_amplitude * self.damping_factor) / (1 + self.damping_factor ** 2)

        partial_g1_t = (-self.gyo_ratio * g2 * self.z0 * self.demagnetization_field -
                        self.gyo_ratio * g3 * self.y0 * self.demagnetization_field -
                        self.gyo_ratio * self.damping_factor * (2 * self.x0 * g1 * self.external_field +
                                                                3 * self.x0 * self.x0 * g1 * self.anisotropy_field +
                                                                self.demagnetization_field * (self.z0 * self.z0 * g1 +
                                                                                              self.x0 * self.z0 * 2 * g3
                                                                                              )) +
                        self.damping_factor * self.gyo_ratio * self.anisotropy_field * g1 -
                        self.stt_amplitude * self.damping_factor * (2 * self.y0 * g2 + 2 * self.z0 * g3) +
                        self.damping_factor * (pow(self.y0, 2) + pow(self.z0, 2)) * stt_partial) / (
                               1 + self.damping_factor ** 2)

        partial_g2_mx = (-self.gyo_ratio * (-self.z0 * self.demagnetization_field + self.z0 * self.anisotropy_field) -
                         self.damping_factor * self.gyo_ratio * (self.y0 * self.external_field +
                                                                 2 * self.x0 * self.y0 * self.anisotropy_field) +
                         self.y0 * self.stt_amplitude * self.damping_factor) / (1 + self.damping_factor ** 2)

        partial_g2_my = (-self.damping_factor * self.gyo_ratio * (self.x0 * self.external_field +
                                                                  self.anisotropy_field * pow(self.x0, 2) +
                                                                  self.demagnetization_field * pow(self.z0, 2)) +
                         self.x0 * self.stt_amplitude * self.damping_factor) / (1 + self.damping_factor ** 2)

        partial_g2_mz = (-self.gyo_ratio * (-self.x0 * self.demagnetization_field +
                                            self.external_field + self.x0 * self.anisotropy_field) -
                         self.damping_factor * self.gyo_ratio * 2 * self.z0 * self.demagnetization_field * self.y0 -
                         self.damping_factor * self.gyo_ratio * self.stt_amplitude) / (1 + self.damping_factor ** 2)

        partial_g2_t = (-self.gyo_ratio * (-self.demagnetization_field * (self.x0 * g3 + self.z0 * g1) +
                                           self.external_field * g3 + self.anisotropy_field * (self.x0 * g3 +
                                                                                               self.z0 * g1)) -
                        self.gyo_ratio * self.damping_factor * (g1 * self.y0 * self.external_field +
                                                                g2 * self.x0 * self.external_field +
                                                                2 * g1 * self.x0 * self.y0 * self.anisotropy_field +
                                                                pow(self.x0, 2) * g2 * self.anisotropy_field +
                                                                self.demagnetization_field * g2 * pow(self.z0, 2) +
                                                                self.demagnetization_field * 2 * self.z0 * g3 * self.y0
                                                                ) -
                        self.damping_factor * self.gyo_ratio * self.stt_amplitude * g3 -
                        self.damping_factor * self.gyo_ratio * stt_partial * self.z0 +
                        self.damping_factor * self.stt_amplitude * (self.x0 * g2 + self.y0 * g1) +
                        self.damping_factor * stt_partial * self.x0 * self.y0) / (1 + self.damping_factor ** 2)

        partial_g3_mx = (self.gyo_ratio * self.y0 * self.anisotropy_field -
                         self.damping_factor * self.gyo_ratio * (
                                 self.external_field + self.x0 * 2 * self.anisotropy_field) * self.z0 +
                         self.damping_factor * self.stt_amplitude * self.z0
                         ) / (1 + self.damping_factor ** 2)

        partial_g3_my = self.gyo_ratio * (self.external_field + self.anisotropy_field * self.x0 +
                                          self.stt_amplitude * self.damping_factor) / (1 + self.damping_factor ** 2)

        partial_g3_mz = (-self.damping_factor * self.gyo_ratio * (self.x0 * self.external_field +
                                                                  self.anisotropy_field * pow(self.x0, 2) +
                                                                  self.demagnetization_field * pow(self.z0, 3) * 3) +
                         self.damping_factor * self.gyo_ratio * self.demagnetization_field +
                         self.stt_amplitude * self.damping_factor * self.x0) / (1 + self.damping_factor ** 2)

        partial_g3_t = (self.gyo_ratio * (
                g2 * self.external_field + self.anisotropy_field * (self.x0 * g2 + g1 * self.y0)) -
                        self.damping_factor * self.gyo_ratio * (g1 * self.z0 * self.external_field +
                                                                g3 * self.x0 * self.external_field +
                                                                self.anisotropy_field * g3 * pow(self.x0, 2) +
                                                                self.anisotropy_field * 2 * g1 * self.z0 * self.x0 +
                                                                self.demagnetization_field * g3 * 3 * pow(self.z0, 2)) +
                        self.damping_factor * self.gyo_ratio * (self.demagnetization_field * g3 +
                                                                self.stt_amplitude * g2) +
                        self.damping_factor * self.gyo_ratio * stt_partial * self.y0 +
                        self.stt_amplitude * self.damping_factor * (g1 * self.z0 + g3 * self.x0) +
                        self.damping_factor * self.x0 * self.z0 * stt_partial) / (
                               1 + self.damping_factor ** 2)

        # initial orthogonal matrix
        if delta_x is None:
            delta_x = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        jacobian_matrix = np.mat([[partial_g1_mx, partial_g1_my, partial_g1_mz, partial_g1_t],
                                  [partial_g2_mx, partial_g2_my, partial_g2_mz, partial_g2_t],
                                  [partial_g3_mx, partial_g3_my, partial_g3_mz, partial_g3_t],
                                  [0, 0, 0, 0]])

        delta_x_interval = np.dot(jacobian_matrix, delta_x)
        # delta_x_new = delta_x + delta_x_interval
        # # Gram-Schmidt Orthogonalization
        # v_matrix, delta_x = gram_schmidt(delta_x_new)

        # return initial states
        [self.x0, self.y0, self.z0] = temp_magnetization
        return delta_x_interval, delta_x

    def lyapunov_exponent(self, calculation_time=10000, cal_t_step=3e-13, dc_amplitude=269, ac_amplitude=0):

        delta_x = None
        h = 6e-13  # time_step of Runge-Kutta

        for i in range(int(calculation_time)):
            # find corresponding magnetization to the current_t
            k1, delta_x1 = self.lyapunov_parameter(current_t=cal_t_step * i, delta_x=delta_x,
                                                   dc_amplitude=dc_amplitude, ac_amplitude=ac_amplitude)

            k2, delta_x2 = self.lyapunov_parameter(current_t=cal_t_step * i + h / 2,
                                                   delta_x=delta_x1 + k1 * h / 2,
                                                   dc_amplitude=dc_amplitude, ac_amplitude=ac_amplitude)

            k3, delta_x3 = self.lyapunov_parameter(current_t=cal_t_step * i + h / 2,
                                                   delta_x=delta_x1 + h / 2 * k2,
                                                   dc_amplitude=dc_amplitude, ac_amplitude=ac_amplitude)

            k4, delta_x4 = self.lyapunov_parameter(current_t=cal_t_step * i + h,
                                                   delta_x=delta_x1 + h * k3,
                                                   dc_amplitude=dc_amplitude, ac_amplitude=ac_amplitude)

            delta_x = delta_x1 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            v_matrix, delta_x = gram_schmidt(delta_x)
            print(i)

            # v_matrix, delta_x = gram_schmidt(delta_x)

            # print('delta_x:{}'.format(delta_x))
            # sum_x = np.log(np.linalg.norm(v_matrix[0, :])) + sum_x
            # sum_y = np.log(np.linalg.norm(v_matrix[1, :])) + sum_y
            # sum_z = np.log(np.linalg.norm(v_matrix[2, :])) + sum_z
            # sum_t = np.log(np.linalg.norm(v_matrix[3, :])) + sum_t

        # calculation of Le
        # sum_x = sum_x / length_n
        # sum_y = sum_y / length_n
        # sum_z = sum_z / length_n
        # sum_t = sum_t / length_n

        # classical method to calculate lyapunov exponent
        temp2 = np.dot(delta_x.T, delta_x)
        print('temp2: {}'.format(temp2))
        eigenvalue_x, _ = np.linalg.eig(temp2)
        lyapunov_exponent = [np.log(abs(i)) / 2 / calculation_time for i in eigenvalue_x]

        return eigenvalue_x, lyapunov_exponent

    def classification_test(self, test_number, node_classification, file_path='weight_matrix_classification',
                            visual_process=False, time_consume_all=1e-8, ac_amplitude=0.0, f_ac=32e9):
        """
        a function used to train weight matrix of readout layer in classification task
        :param file_path: the path of weight_matrix
        :param test_number: number of wave forms
        :param node_classification: Size of reservoirs
        :param visual_process: showing the process figures if it is True, default as False
        :param time_consume_all: time consume in single step evolution
        :param ac_amplitude: amplitude of ac term
        :param f_ac: frequency of ac stt term
        """

        if os.path.exists(f'{file_path}/classification_ac_{ac_amplitude}_node_{node_classification}.npy'):
            weight_out_stm = np.load(f'{file_path}/classification_ac_{ac_amplitude}_node_{node_classification}.npy')
            print('###############################################')
            print(f'{file_path}/classification_ac_{ac_amplitude}_node_{node_classification}.npy loading !')
            print('###############################################')

        else:
            sys.exit(r'No weight data!')

        # fabricate the input, target, and output signals
        y_out_list = []
        s_in, train_signal = waveform_generator(test_number)

        # pre training
        self.time_evolution(dc_amplitude=100, ac_amplitude=ac_amplitude,
                            time_consumed=time_consume_all, f_ac=f_ac)

        mz_list_eight_points = []

        for i in track(range(len(s_in))):
            _, _, mz_list_chao, t_list2 = self.time_evolution(dc_amplitude=s_in[i], ac_amplitude=ac_amplitude,
                                                              time_consumed=time_consume_all, f_ac=f_ac)
            mz_list_eight_points = np.append(mz_list_eight_points, mz_list_chao)

            if (i + 1) % 8 == 0:
                try:
                    mz_list_all = mz_list_eight_points[argrelmax(mz_list_eight_points)]
                    xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
                    fp = mz_list_all
                    sampling_x_values = np.linspace(1, len(mz_list_all), node_classification)
                    # linear slinear quadratic cubic
                    f = interp1d(xp, fp, kind='quadratic')
                    x_matrix1 = f(sampling_x_values)

                except Exception as ErrorMessage:
                    print('Error in sampling')
                    sys.exit(ErrorMessage)

                # add bias term
                try:
                    x_matrix1 = np.append(x_matrix1.T, 1).reshape(-1, 1)
                    y_out = np.dot(weight_out_stm, x_matrix1)
                    y_out_list.append(y_out[0, 0])

                except ValueError as error:
                    print('----------------------------------------')
                    print('error from readout layer: {}'.format(error))
                    print('Please check for your weight file at {}'.format(file_path))
                    print('________________________________________')
                    return 0

        output_result = np.where(np.array(y_out_list) > 0.5, 1, 0)
        result_compare = (train_signal == output_result)
        accuracy = len(np.argwhere(result_compare)) / len(result_compare)
        print(f'accuracy: {accuracy}%')

        # visualization of magnetization
        if visual_process:
            plt.figure('output')
            plt.title('real output')
            plt.plot(output_result)
            plt.ylabel('Output signals')
            plt.xlabel('Time')

            plt.figure('target output')
            plt.title('target signals')
            plt.plot(train_signal)
            plt.ylabel('target signals')
            plt.xlabel('Time')

            plt.figure('original output')
            plt.title('Original Output')
            plt.plot(y_out_list)
            plt.ylabel('Output signals')
            plt.xlabel('Time')
            plt.show()

        return accuracy

    def classification_train(self, number_wave, node_classification, file_path='weight_matrix_classification',
                             visual_process=False, save_index=True, time_consume_all=1e-8, ac_amplitude=0.0,
                             f_ac=32e9, recover_weight=False):
        """
        a function used to train weight matrix of readout layer in classification task
        :param file_path: the path of weight_matrix
        :param number_wave: number of wave forms
        :param node_classification: Size of reservoirs
        :param visual_process: showing the process figures if it is True, default as False
        :param save_index: save weight matrix file or not
        :param time_consume_all: time consume in single step evolution
        :param ac_amplitude: amplitude of ac term
        :param f_ac: frequency of ac stt term
        :param recover_weight: boolean, recover all of weight data if True
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if os.path.exists(f'{file_path}/classification_ac_{ac_amplitude}_node_{node_classification}.npy'):
            weight_out_stm = np.load(f'{file_path}/classification_ac_{ac_amplitude}_node_{node_classification}.npy')
            print('###############################################')
            print(f'{file_path}/classification_ac_{ac_amplitude}_node_{node_classification}.npy already exists !')
            print('###############################################')

            if not recover_weight:
                return 0
            else:
                print('NO MATTER to retrain again !')

        else:
            # think about bias term
            weight_out_stm = np.random.randint(-1, 2, (1, node_classification + 1))
            print('\r weight matrix of classification task created successfully !', end='', flush=True)

        # fabricate the input, target, and output signals
        y_out_list, x_final_matrix = [], []
        trace_reservoirs, trace_mz = [], []

        s_in, train_signal = waveform_generator(number_wave)

        # pre training
        self.time_evolution(dc_amplitude=100, ac_amplitude=ac_amplitude,
                            time_consumed=time_consume_all, f_ac=f_ac)

        mz_list_eight_points = []

        for i in track(range(len(s_in))):
            _, _, mz_list_chao, t_list2 = self.time_evolution(dc_amplitude=s_in[i], ac_amplitude=ac_amplitude,
                                                              time_consumed=time_consume_all, f_ac=f_ac)
            mz_list_eight_points = np.append(mz_list_eight_points, mz_list_chao)
            trace_mz = np.append(trace_mz, mz_list_chao)

            if (i + 1) % 8 == 0:
                try:
                    mz_list_all = mz_list_eight_points[argrelmax(mz_list_eight_points)]
                    trace_reservoirs = np.append(trace_reservoirs, mz_list_all)
                    xp = np.linspace(1, len(mz_list_all), len(mz_list_all))
                    fp = mz_list_all
                    sampling_x_values = np.linspace(1, len(mz_list_all), node_classification)
                    # linear slinear quadratic cubic
                    f = interp1d(xp, fp, kind='quadratic')
                    x_matrix1 = f(sampling_x_values)

                except Exception as ErrorMessage:
                    print('Error in sampling')
                    sys.exit(ErrorMessage)

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
        weight_out_stm = np.dot(y_train_matrix, np.linalg.pinv(x_final_matrix))

        y_test = np.dot(weight_out_stm, x_final_matrix)
        print('train result:{}'.format(y_test))

        print('##################################################################')
        print('Trained successfully !')
        print('##################################################################')

        # save weight matrix as .npy files
        if save_index:
            np.save(f'{file_path}/classification_ac_{ac_amplitude}_node_{node_classification}.npy', weight_out_stm)
            print('Saved weight matrix file')

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
            plt.ylabel(r'$m_z$')
            t1 = np.linspace(0, len(trace_reservoirs) - 1, len(trace_reservoirs))
            plt.scatter(t1, trace_reservoirs)

            plt.figure('input')
            plt.title('input signals')
            plt.plot(s_in)
            plt.ylabel('inputs')
            plt.xlabel('Time')
            plt.show()

    def real_time_train(self, number_wave, nodes_stm, file_path='weight_matrix_oscillator_xuezhao',
                        visual_process=False,
                        save_index=True, alert_index=False, superposition=0, time_consume_all=1e-8, ac_amplitude=0.0,
                        f_ac=32e9, recover_weight=False, task='Delay'):
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
        :param recover_weight: boolean, recover all of weight data if True
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
            print('\r weight matrix of STM created successfully !', end='', flush=True)

        # fabricate the input, target, and output signals
        y_out_list, x_final_matrix, plus_visual_mz, minus_visual_mz = [], [], [], []
        plus_time, minus_time, time_index = [], [], 0

        # it seems to build a function better
        positive_dc_current = 200
        negative_dc_current = 100

        s_in, train_signal = real_time_generator(task=f'{task}', superposition_number=superposition,
                                                 length_signals=number_wave)
        # pre training
        self.time_evolution(dc_amplitude=positive_dc_current, ac_amplitude=ac_amplitude,
                            time_consumed=time_consume_all,
                            f_ac=f_ac)

        trace_mz = []

        for i in track(range(len(s_in))):
            if s_in[i] == 1:
                dc_current1 = positive_dc_current
                _, _, mz_list_chao, t_list2 = self.time_evolution(dc_amplitude=dc_current1, ac_amplitude=ac_amplitude,
                                                                  time_consumed=time_consume_all, f_ac=f_ac)

            else:
                dc_current1 = negative_dc_current
                _, _, mz_list_chao, t_list2 = self.time_evolution(dc_amplitude=dc_current1, ac_amplitude=ac_amplitude,
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
        print('train result:{}'.format(y_test))

        # calculate the error
        error_learning = np.var(np.array(train_signal) - np.array(y_test))
        print('##################################################################')
        print('error:{}'.format(error_learning))
        print('Trained successfully !')
        print('##################################################################')

        # save weight matrix as .npy files
        if save_index:
            np.save(f'{file_path}/STM_{task}_{superposition}_node_{nodes_stm}_ac_{ac_amplitude}.npy', weight_out_stm)
            print('Saved weight matrix file')

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
            plt.scatter(minus_time, minus_visual_mz, label='oscillator', s=2, c='red')
            plt.scatter(plus_time, plus_visual_mz, label='{}'.format(positive_dc_current), s=2, c='blue')
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
                       task='Delay'):
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
            return 0

        # fabricate the input, target, and output signals
        y_out_list, x_final_matrix, plus_visual_mz, minus_visual_mz = [], [], [], []
        plus_time, minus_time, time_index = [], [], 0

        # it seems to build a function better
        positive_dc_current = 200
        negative_dc_current = 100

        s_in, train_signal = real_time_generator(task=f'{task}', superposition_number=superposition,
                                                 length_signals=test_number)

        trace_mz = []
        # pre training
        self.time_evolution(dc_amplitude=positive_dc_current, ac_amplitude=ac_amplitude,
                            f_ac=f_ac, time_consumed=time_consume_all)
        # time_consume_all = 1e-8

        for i_1 in track(range(len(s_in))):
            if s_in[i_1] == 1:
                dc_current1 = positive_dc_current
                _, _, mz_list_chao, t_list2 = self.time_evolution(dc_amplitude=dc_current1, ac_amplitude=ac_amplitude,
                                                                  f_ac=f_ac, time_consumed=time_consume_all)

            else:
                dc_current1 = negative_dc_current
                _, _, mz_list_chao, t_list2 = self.time_evolution(dc_amplitude=dc_current1, ac_amplitude=ac_amplitude,
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
                y_out_list.append(y_out[0, 0])
                x_final_matrix.append(x_matrix1.T.tolist()[0])

            except ValueError as error:
                print('----------------------------------------')
                print('error from readout layer: {}'.format(error))
                print('Please check for your weight file at {}'.format(file_path))
                print('________________________________________')
                return 0

        # calculate the error
        # error_learning = np.var(np.array(train_signal) - np.array(y_out_list))
        error_learning = (
            np.square(np.array(train_signal[superposition:]) - np.array(y_out_list)[superposition:])).mean()
        print('Test Error:{}'.format(error_learning))
        print('----------------------------------------------------------------')

        capacity = pow(np.corrcoef(y_out_list[superposition:], train_signal[superposition:])[0, 1], 2)

        if alert_index:
            email_alert(subject='error of STM : {}'.format(error_learning))

        # FIGURES
        if visual_index:
            plt.figure('Test results')
            plt.plot(train_signal, c='blue', label='target')
            plt.plot(y_out_list, c='green', label='module')
            plt.ylabel('signals')
            plt.xlabel('Time')
            plt.legend()

            plt.figure('Comparison')
            plt.subplot(2, 1, 1)
            plt.title('Mean square error : {}'.format(error_learning))
            plt.plot(train_signal, c='blue', label='target output')
            plt.legend()
            plt.ylabel('signals')
            plt.xlabel('Time')

            plt.subplot(2, 1, 2)
            plt.plot(y_out_list, c='red', label='actual output')
            # plt.title('Mean square error : {}'.format(error_learning))
            plt.legend()
            plt.ylabel('signals')
            plt.xlabel('Time')
            plt.show()

        return capacity
