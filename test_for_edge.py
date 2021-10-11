import os
import sys
import time
import scipy
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.header import Header
from rich.progress import track


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
    for i in random_pulse:
        if i == 0:
            # sine function
            sine_points = [268, 268.5, 269, 268.5, 268, 267.5, 267, 267.5]
            # sine_points = [268, 268.5, 269, 268.5, 268, 268.5, 267, 268.5]
            wave_points = wave_points + sine_points
        elif i == 1:
            # square function
            square_points = [269] * 4 + [267] * 4
            wave_points = wave_points + square_points

    print('wave:{}'.format(wave_points))

    # FIGURE
    plt.figure('input-signals')
    wave_signals = []
    for j1 in wave_points:
        temp2 = [j1] * 20
        wave_signals = wave_signals + temp2
    temp1 = np.linspace(0, len(wave_signals) - 1, len(wave_signals))
    plt.plot(temp1, wave_signals)
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
                for i in range(1, superposition_number + 1):
                    temp_signal = np.append(s_in[-int(i):], s_in[:-int(i)])
                    train_signal = train_signal + temp_signal
                    train_signal[np.argwhere(train_signal == 2)] = 0

        print('############################################################')
        print('inputs :{}'.format(s_in))
        print('target :{}'.format(train_signal))
        print('############################################################')

        return s_in, train_signal

    except Exception as error:
        print('Sth wrong in generating input signals:{}'.format(error))
        sys.exit(0)


def edge_of_chaos(initial_dif=1e-8, len_input_number=10, time_consume_single=7e-9, len_input_pattern=10,
                  ac_current1=0):
    """
    a function used to find edge of chaos, target is magnetization itself not reservoirs
    :param initial_dif: the initial difference / perturbation
    :param len_input_number:  the number of calculation times over different inputs sequence
    :param time_consume_single:  time used to make evolution of magnetization
    :param len_input_pattern:  lengthen of single input signal
    :param ac_current1: amplitude of ac current, unit: Oe
    :return: largest lyapunov exponent
    """

    positive_dc_current = 100
    negative_dc_current = 200
    delta_t_diff = []

    t_step = 3e-13
    print('Time lengthen : {}'.format(int(time_consume_single / t_step)))

    for i in range(len_input_number):
        t1 = time.time()
        trace_perturbation = Mtj()
        mtj2 = Mtj()
        # trace1 and trace2 in attractor
        trace_perturbation.time_evolution(time_consumed=1e-8, dc_amplitude=positive_dc_current,
                                          ac_amplitude=ac_current1)
        mtj2.time_evolution(time_consumed=1e-8, dc_amplitude=positive_dc_current,
                            ac_amplitude=ac_current1)
        # introduce a perturbation
        trace_perturbation.z0 += initial_dif
        mode = np.sqrt(pow(trace_perturbation.x0, 2) + pow(trace_perturbation.y0, 2) + pow(trace_perturbation.z0, 2))
        trace_perturbation.x0, trace_perturbation.z0 = trace_perturbation.x0 / mode, trace_perturbation.z0 / mode
        trace_perturbation.y0 = trace_perturbation.y0 / mode
        initial_dif_real = np.sqrt(
            pow(trace_perturbation.x0 - mtj2.x0, 2) + pow(trace_perturbation.y0 - mtj2.y0, 2) + pow(
                trace_perturbation.z0 - mtj2.z0, 2))
        # print('initial perturbation : {}'.format(initial_dif_real))

        s_in = np.random.randint(0, 2, len_input_pattern)
        single_delta_t = []
        # print('single input:{}'.format(s_in))
        for j in s_in:
            if j == 1:
                dc_current1 = positive_dc_current
            elif j == 0:
                dc_current1 = negative_dc_current

            trace_perturbation.time_evolution(time_consumed=time_consume_single, dc_amplitude=dc_current1,
                                              ac_amplitude=ac_current1)
            mtj2.time_evolution(time_consumed=time_consume_single, dc_amplitude=dc_current1, ac_amplitude=ac_current1)

            # calculate error and reset
            diff_temp = np.sqrt(
                pow((trace_perturbation.x0 - mtj2.x0), 2) + pow(trace_perturbation.y0 - mtj2.y0, 2) + pow(
                    trace_perturbation.z0 - mtj2.z0, 2))
            trace_perturbation.x0 = mtj2.x0 + initial_dif_real / diff_temp * (trace_perturbation.x0 - mtj2.x0)
            trace_perturbation.y0 = mtj2.y0 + initial_dif_real / diff_temp * (trace_perturbation.y0 - mtj2.y0)
            trace_perturbation.z0 = mtj2.z0 + initial_dif_real / diff_temp * (trace_perturbation.z0 - mtj2.z0)
            mode = np.sqrt(
                pow(trace_perturbation.x0, 2) + pow(trace_perturbation.y0, 2) + pow(trace_perturbation.z0, 2))
            trace_perturbation.x0, trace_perturbation.z0 = trace_perturbation.x0 / mode, trace_perturbation.z0 / mode
            trace_perturbation.y0 = trace_perturbation.y0 / mode
            single_delta_t.append(np.log(diff_temp / initial_dif_real))
            # print('each log d1/d0 : {}'.format(single_delta_t))

        delta_t_diff.append(np.mean(single_delta_t))
        t2 = time.time()
        print('delta_t_: {}  ac_current: {}  time_used: {:.3f}s'.format(delta_t_diff[i], ac_current1, t2 - t1))
        print('##################################################################')

    # calculate lyapunov exponent
    le_z = np.mean(delta_t_diff)
    print('******************************************************************')
    print('le_z : {}'.format(le_z))
    print('delta_t_diff : {}'.format(delta_t_diff))
    print('******************************************************************')
    return le_z, delta_t_diff


def chaos_mine(initial_dif=1e-8, time_consume_single=1e-8, ac_current1=0):
    """
    a function used to find edge of chaos, calculation should start at reservoirs rather than magnetization.
    :param initial_dif: the initial difference / perturbation
    :param time_consume_single:  time used to make evolution of magnetization
    :param ac_current1: amplitude of ac current, unit: Oe
    :return: largest lyapunov exponent
    """

    # hyper parameters
    positive_dc_current = 100
    negative_dc_current = 200
    time_step = 3e-13

    # random initial state to test my algorithm
    initial_state = np.random.random(3)

    # enter into an orbit
    trace = Mtj(initial_state[0], initial_state[1], initial_state[2])
    trace_perturbation = Mtj(initial_state[0], initial_state[1], initial_state[2])
    state = trace.get_reservoirs(dc_current=positive_dc_current, ac_current=ac_current1,
                                 consuming_time=time_consume_single, size=16)
    state_perturbation = trace_perturbation.get_reservoirs(dc_current=positive_dc_current, ac_current=ac_current1,
                                                           consuming_time=time_consume_single, size=16)

    # adding perturbation
    state_perturbation[-1, :] = state_perturbation[-1, :] + np.array([0, 0, initial_dif])
    state_perturbation[-1, :] = state_perturbation[-1, :] / np.linalg.norm(state_perturbation[-1, :])
    real_initial_dif = np.linalg.norm(state_perturbation[-1, :] - state[-1, :])

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
    for i1 in range(iteration_number):
        trace.m = state[-1, :]
        trace_perturbation.m = state_perturbation[-1, :]

        # random input dc amplitude
        s_in = np.random.randint(0, 2, 1)[0]
        if s_in == 1:
            dc_current_input = positive_dc_current
        elif s_in == 0:
            dc_current_input = negative_dc_current
        state = trace.get_reservoirs(dc_current=dc_current_input, ac_current=ac_current1,
                                     consuming_time=time_consume_single, size=16)
        state_perturbation = trace_perturbation.get_reservoirs(dc_current=dc_current_input, ac_current=ac_current1,
                                                               consuming_time=time_consume_single, size=16)
        # here only calculate the last reservoir state
        difference_trace = np.linalg.norm(state_perturbation[-1, :] - state[-1, :])
        if difference_trace == 0:
            print('Error : the initial difference is too small')
            sys.exit()

        le = np.log(abs(difference_trace / real_initial_dif))
        le_buffer.append(le)

        # reset trace_perturbation to avoid numerical overflow
        state_perturbation[-1, :] = state[-1, :] + real_initial_dif / difference_trace * (state_perturbation[-1,
                                                                                          :] - state[-1, :])
        state_perturbation[-1, :] = state_perturbation[-1, :] / np.linalg.norm(state_perturbation[-1, :])

        # info
        print('##################################### info ########################################')
        print('State : {}  State_perturbation: {}'.format(state[-1, :], state_perturbation[-1, :]))
        print('Epoch: {} Current value : {} Average value: {}'.format(i1 + 1, le, np.mean(le_buffer)))

        if i1 == 100:
            le_buffer = []
            print('clear buffer')


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

    def step_evolution(self, time_ac, magnetization, dc_amplitude, ac_amplitude):
        # time evolution from differential equation
        self.m = magnetization/np.linalg.norm(magnetization)
        self.x0, self.y0, self.z0 = self.m
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

    def time_evolution(self, dc_amplitude=420.21, ac_amplitude=0, time_consumed=1e-8):
        sol = scipy.integrate.solve_ivp(self.step_evolution, t_span=(0, time_consumed), y0=self.m,
                                        t_eval=np.linspace(0, time_consumed, int(time_consumed / self.t_step)),
                                        args=[dc_amplitude, ac_amplitude])
        t_list = sol.t
        mx_list, my_list, mz_list = sol.y
        # normalization
        norms = np.linalg.norm(sol.y, axis=0)
        mx_list, my_list, mz_list = mx_list/norms, my_list/norms, mz_list/norms
        self.x0, self.y0, self.z0 = mx_list[-1], my_list[-1], mz_list[-1]
        self.m = np.array([self.x0, self.y0, self.z0])
        return mx_list, my_list, mz_list, t_list

    def get_reservoirs(self, dc_current=100, ac_current=0, consuming_time=1e-8, size=16):
        mx_list, my_list, mz_list, _ = self.time_evolution(dc_amplitude=dc_current, ac_amplitude=ac_current,
                                                           time_consumed=consuming_time)
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

    def classification_train(self, number_wave, nodes_classification):
        """
        a function used to train weight matrix of readout layer in classification task
        :param number_wave: number of wave forms
        :param nodes_classification: Size of reservoirs
        """

        if os.path.exists('weight_matrix_chaos/weight_out_classification.npy'):
            weight_out_stm = np.load('weight_matrix_chaos/weight_out_classification.npy')
            print('\r' + 'Loading weight_out_classification matrix successfully !', end='', flush=True)

        else:
            # think about bias term
            weight_out_stm = np.random.randint(-1, 2, (1, nodes_classification + 1))
            print('\r weight matrix of STM created successfully !', end='', flush=True)

        print('\rClassification')
        print('----------------------------------------------------------------')
        print('start to train !', flush=True)

        # fabricate the input, target, and output signals
        y_out_list, x_final_matrix, resistance_dif_mat = [], [], []
        s_in, train_signal = waveform_generator(number_wave)
        print('---------------------------------------------------------------')

        for i in track(range(len(s_in))):
            mx_list2, my_list2, mz_list2, t_list2 = self.time_evolution(dc_amplitude=s_in[i], ac_amplitude=20,
                                                                        time_consumed=1e-8)

            try:
                max_extreme1, min_extreme1, resistance_dif_list1 = [], [], []
                for i1 in range(len(mz_list2)):
                    if i1 != 0 and i1 != len(mz_list2) - 1:
                        if mz_list2[i1] > mz_list2[i1 - 1] and mz_list2[i1] > mz_list2[i1 + 1]:
                            max_extreme1.append(mz_list2[i])
                        elif mz_list2[i1] < mz_list2[i1 - 1] and mz_list2[i1] < mz_list2[i1 + 1]:
                            min_extreme1.append(mz_list2[i1])

                length_extreme1 = min(len(max_extreme1), len(min_extreme1))
                print('length of extreme points:{}'.format(length_extreme1))

                if length_extreme1 > 25:
                    length_extreme1 = 25
                else:
                    print('length of sampling consequence is error')

                for i1 in range(length_extreme1):
                    resistance_dif_list1.append(max_extreme1[i1] - min_extreme1[i1])

                print('resist: {}'.format(resistance_dif_list1))
                resistance_dif_mat = resistance_dif_mat + resistance_dif_list1

            except Exception as error:
                print('error in finding max_extreme1 or min_extreme: {}'.format(error))
                print('max_point :{}'.format(len(max_extreme1)))
                print('min_point :{}'.format(len(min_extreme1)))

            # sampling points
            if (i + 1) % 8 == 0:
                number_interval = int(len(resistance_dif_mat) / nodes_classification)
                if number_interval < 1:
                    number_interval = 1
                x_matrix1 = np.array(resistance_dif_mat[1: len(resistance_dif_mat):number_interval])
                resistance_dif_mat = []

                while len(x_matrix1) != nodes_classification:
                    if len(x_matrix1) > nodes_classification:
                        x_matrix1 = x_matrix1[:-1]
                    else:
                        x_matrix1 = np.append(x_matrix1, x_matrix1[-1])
                x_matrix1 = np.reshape(x_matrix1, (nodes_classification, 1))

                # add bias term
                x_matrix1 = np.append(x_matrix1.T, 1).reshape(-1, 1)
                y_out = np.dot(weight_out_stm, x_matrix1)
                y_out_list.append(y_out[0, 0])
                x_final_matrix.append(x_matrix1.T.tolist()[0])

            # calculation time
            if i % 20 == 0:
                print('-------------------------------------------------------------')
                print('progress : {:.3} % classification training'.format(i / len(s_in) * 100))
                print('-------------------------------------------------------------')

        # update weight
        y_train_matrix = np.array(train_signal).reshape(1, len(train_signal))
        x_final_matrix = np.asmatrix(x_final_matrix).T
        weight_out_stm = np.dot(y_train_matrix, np.linalg.pinv(x_final_matrix))

        # save weight matrix as .npy files
        np.save('weight_matrix_chaos/weight_out_classification.npy', weight_out_stm)

        # calculate the error
        error_learning = np.var(np.array(train_signal) - np.array(y_out_list))
        print('error:{}'.format(error_learning))
        print('Trained successfully !')
        print('----------------------------------------------------------------')
        email_alert(subject='Training Successfully !')

    def classification_test(self, test_number=80, nodes_classification=80):
        """
        a function used to test the ability of classification of chaotic-MTJ echo state network
        :param test_number: the number of test waves form, default:80
        :param nodes_classification: Size of classification, default:80
        """

        if os.path.exists('weight_matrix_chaos/weight_out_classification.npy'):
            weight_out_stm = np.load('weight_matrix_chaos/weight_out_classification.npy')
            print('\r' + 'Loading weight_out_classification matrix successfully !', end='', flush=True)
            print(weight_out_stm)

        else:
            print('\rno valid weight data !', end='', flush=True)
            return 0

        print('\rClassification')
        print('----------------------------------------------------------------')
        print('start to test !', flush=True)

        # fabricate the input, target, and output signals
        y_out_list, x_final_matrix, resistance_dif_mat = [], [], []
        s_in, train_signal = waveform_generator(test_number)
        print('---------------------------------------------------------------')

        for i in track(range(len(s_in))):
            mx_list2, my_list2, mz_list2, t_list2 = self.time_evolution(dc_amplitude=s_in[i], ac_amplitude=20,
                                                                        time_consumed=1e-8)

            try:
                max_extreme1, min_extreme1, resistance_dif_list1 = [], [], []
                for i1 in range(len(mz_list2)):
                    if i1 != 0 and i1 != len(mz_list2) - 1:
                        if mz_list2[i1] > mz_list2[i1 - 1] and mz_list2[i1] > mz_list2[i1 + 1]:
                            max_extreme1.append(mz_list2[i])
                        elif mz_list2[i1] < mz_list2[i1 - 1] and mz_list2[i1] < mz_list2[i1 + 1]:
                            min_extreme1.append(mz_list2[i1])

                length_extreme1 = min(len(max_extreme1), len(min_extreme1))
                print('length of extreme points:{}'.format(length_extreme1))

                # ensure the sampling points
                if length_extreme1 > 25:
                    length_extreme1 = 25
                else:
                    print('length of consequence is error')

                for i1 in range(length_extreme1):
                    resistance_dif_list1.append(max_extreme1[i1] - min_extreme1[i1])
                resistance_dif_mat = resistance_dif_mat + resistance_dif_list1

            except Exception as error:
                print('error in finding max_extreme1 or min_extreme: {}'.format(error))
                print('max_point :{}'.format(len(max_extreme1)))
                print('min_point :{}'.format(len(min_extreme1)))

            # sampling points
            if (i + 1) % 8 == 0:

                if length_extreme1 > nodes_classification:
                    print('the nodes number is too large ')

                number_interval = int(len(resistance_dif_mat) / nodes_classification)
                if number_interval < 1:
                    number_interval = 1
                x_matrix1 = np.array(resistance_dif_mat[1: len(resistance_dif_mat):number_interval])
                resistance_dif_mat = []

                while len(x_matrix1) != nodes_classification:
                    if len(x_matrix1) > nodes_classification:
                        x_matrix1 = x_matrix1[:-1]
                    else:
                        x_matrix1 = np.append(x_matrix1, x_matrix1[-1])
                x_matrix1 = np.reshape(x_matrix1, (nodes_classification, 1))

                # add bias term
                x_matrix1 = np.append(x_matrix1.T, 1).reshape(-1, 1)
                y_out = np.dot(weight_out_stm, x_matrix1)
                print('y_out :{}'.format(y_out))
                y_out_list.append(y_out[0, 0])

            print('-------------------------------------------------------------')
            print('progress : {} % classification testing'.format(i / len(s_in) * 100))
            print('-------------------------------------------------------------')

        # calculate the error
        error_learning = np.var(np.array(train_signal) - np.array(y_out_list))
        print('Test Error:{}'.format(error_learning))
        print('----------------------------------------------------------------')

        # FIGURES
        plt.figure('Test results')
        plt.plot(train_signal, c='blue', label='target')
        plt.plot(y_out_list, c='green', label='module')
        plt.ylabel('Index')
        plt.xlabel('Time')
        plt.legend()
        # plt.show()

        plt.figure('Comparison')
        plt.subplot(2, 1, 1)
        plt.plot(train_signal, c='blue', label='target & input')
        plt.legend()
        plt.ylabel('Index')
        plt.xlabel('Time')

        plt.subplot(2, 1, 2)
        plt.plot(y_out_list, c='red', label='module')
        plt.legend()
        plt.ylabel('Index')
        plt.xlabel('Time')
        plt.show()

    def stm_train(self, number_wave, nodes_stm, file_path='weight_matrix_oscillator_xuezhao', visual_process=False,
                  save_index=True, alert_index=False, superposition=0, time_consume_all=1e-8, ac_amplitude=0):
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
        :param ac_amplitude: amplitude of ac stt term
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if os.path.exists('{}/STM_delay_{}_node_{}.npy'.format(file_path, superposition, nodes_stm)):
            weight_out_stm = np.load('{}/STM_delay_{}_node_{}.npy'.format(file_path, superposition, nodes_stm))
            print('###############################################')
            print('Loading weight matrix successfully !')
            print('shape of weight:{}'.format(weight_out_stm.shape))
            print('###############################################')

        else:
            # think about bias term
            weight_out_stm = np.random.randint(-1, 2, (1, nodes_stm + 1))
            print('\r weight matrix of STM created successfully !', end='', flush=True)

        # print('\r short term memory')
        # print('----------------------------------------------------------------')
        # print('start to train !', flush=True)

        # fabricate the input, target, and output signals
        y_out_list, x_final_matrix, plus_visual_mz, minus_visual_mz = [], [], [], []
        plus_time, minus_time, time_index = [], [], 0

        # it seems to build a function better
        positive_dc_current = 200
        negative_dc_current = 100

        s_in, train_signal = real_time_generator(task='Delay', superposition_number=superposition,
                                                 length_signals=number_wave)

        # print('---------------------------------------------------------------')
        trace_mz = []

        for i in track(range(len(s_in))):
            if s_in[i] == 1:
                dc_current1 = positive_dc_current
                _, _, mz_list, t_list2 = self.time_evolution(
                    dc_amplitude=dc_current1, ac_amplitude=ac_amplitude, time_consumed=time_consume_all)

            else:
                dc_current1 = negative_dc_current
                _, _, mz_list, t_list2 = self.time_evolution(
                    dc_amplitude=dc_current1, ac_amplitude=ac_amplitude, time_consumed=time_consume_all)

            try:
                mz_list_all, t_list_whole = [], []
                extreme_high, extreme_low = [], []
                for i1 in range(len(mz_list)):
                    if i1 != 0 and i1 != len(mz_list) - 1:
                        if mz_list[i1] > mz_list[i1 - 1] and mz_list[i1] > mz_list[i1 + 1]:
                            extreme_high.append(mz_list[i1])
                        if mz_list[i1] < mz_list[i1 - 1] and mz_list[i1] < mz_list[i1 + 1]:
                            extreme_low.append(mz_list[i1])

                length_extreme = min(len(extreme_low), len(extreme_high))
                for i2 in range(length_extreme):
                    mz_list_all.append(extreme_high[i2])

            except Exception as error:
                print('error from sampling curve :{}'.format(error))
                mz_list_all = mz_list

            # print('mz_list_all:{}'.format(mz_list_all))
            # print('len of mz sampling points :{}'.format(len(mz_list_all)))
            trace_mz = trace_mz + mz_list_all

            # for figures
            if s_in[i] == 1:
                plus_visual_mz = plus_visual_mz + list(mz_list)
                plus_time = plus_time + list(
                    np.linspace(time_index, time_index + 3e-13 * (len(t_list2) - 1), len(t_list2)))
            else:
                minus_visual_mz = minus_visual_mz + list(mz_list)
                minus_time = minus_time + list(
                    np.linspace(time_index, time_index + 3e-13 * (len(t_list2) - 1), len(t_list2)))

            time_index = time_index + t_list2[-1]

            # sampling points
            try:
                if len(mz_list_all) < nodes_stm:
                    print('the nodes number is too large')
                    print('len of mz_list_all : {}'.format(len(mz_list_all)))
                    print('length of nodes number : {}'.format(nodes_stm))
                    return 0

                number_interval = int(len(mz_list_all) / nodes_stm)
                final_numbers = number_interval * nodes_stm
                x_matrix1 = np.array(mz_list_all[0: final_numbers: number_interval])

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
            np.save('{}/STM_delay_{}_node_{}.npy'.format(file_path, superposition, nodes_stm), weight_out_stm)
            print('Saved weight matrix file')

        # visualization of magnetization
        if visual_process:
            plt.figure('Trace of mz')
            plt.title('reservoirs states')
            plt.xlabel('Time interval')
            plt.ylabel(r'$m_z$')
            # t1 = np.linspace(0, len(trace_mz) - 1, len(trace_mz))
            plt.plot(trace_mz)
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

    def stm_test(self, test_number=80, nodes_stm=80, file_path='weight_matrix_oscillator_xuezhao', visual_index=True,
                 alert_index=False, superposition=0, time_consume_all=1e-8, ac_amplitude=0):
        """
        a function used to test the ability of classification of chaotic-MTJ echo state network
        :param test_number: the number of test waves form, default:80
        :param nodes_stm: Size of classification, default:80
        :param file_path: Path of weight matrix file
        :param visual_index: show test result as figures if it is True
        :param alert_index: sending notification when it is True
        :param superposition: number of time interval or delay time
        :param time_consume_all: time consume in single step evolution
        :param ac_amplitude: amplitude of ac stt term
        """

        if os.path.exists('{}/STM_delay_{}_node_{}.npy'.format(file_path, superposition, nodes_stm)):
            weight_out_stm = np.load('{}/STM_delay_{}_node_{}.npy'.format(file_path, superposition, nodes_stm))
            print('Loading STM_delay_{}_node_{} matrix successfully !'.format(superposition, nodes_stm))
            print('shape of weight:{}'.format(weight_out_stm.shape))
            # print('weight: {}'.format(weight_out_stm))
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

        s_in, train_signal = real_time_generator(task='Delay', superposition_number=superposition,
                                                 length_signals=test_number)

        trace_mz = []

        # time_consume_all = 1e-8

        for i_1 in track(range(len(s_in))):
            if s_in[i_1] == 1:
                dc_current1 = positive_dc_current
                _, _, mz_list, t_list2 = self.time_evolution(
                    dc_amplitude=dc_current1, ac_amplitude=ac_amplitude, time_consumed=time_consume_all)

            else:
                dc_current1 = negative_dc_current
                _, _, mz_list, t_list2 = self.time_evolution(
                    dc_amplitude=dc_current1, ac_amplitude=ac_amplitude, time_consumed=time_consume_all)

            try:
                mz_list_all, t_list_whole = [], []
                extreme_high, extreme_low = [], []
                for i1 in range(len(mz_list)):
                    if i1 != 0 and i1 != len(mz_list) - 1:
                        if mz_list[i1] > mz_list[i1 - 1] and mz_list[i1] > mz_list[i1 + 1]:
                            extreme_high.append(mz_list[i1])
                        if mz_list[i1] < mz_list[i1 - 1] and mz_list[i1] < mz_list[i1 + 1]:
                            extreme_low.append(mz_list[i1])

                length_extreme = min(len(extreme_low), len(extreme_high))
                for i2 in range(length_extreme):
                    mz_list_all.append(extreme_high[i2])

            except Exception as error:
                print('error from sampling curve :{}'.format(error))
                mz_list_all = mz_list

            # print('mz_list_all:{}'.format(mz_list_all))
            # print('len of mz sampling points :{}'.format(len(mz_list_all)))
            trace_mz = trace_mz + mz_list_all

            # for figures
            if s_in[i_1] == 1:
                plus_visual_mz = plus_visual_mz + list(mz_list)
                plus_time = plus_time + list(
                    np.linspace(time_index, time_index + 3e-13 * (len(t_list2) - 1), len(t_list2)))
            else:
                minus_visual_mz = minus_visual_mz + list(mz_list)
                minus_time = minus_time + list(
                    np.linspace(time_index, time_index + 3e-13 * (len(t_list2) - 1), len(t_list2)))

            time_index = time_index + t_list2[-1]

            # sampling points
            try:
                if len(mz_list_all) < nodes_stm:
                    print('the nodes number is too large')
                    print('len of mz_list_all : {}'.format(len(mz_list_all)))
                    print('length of nodes number : {}'.format(nodes_stm))
                    return 0

                number_interval = int(len(mz_list_all) / nodes_stm)
                final_numbers = number_interval * nodes_stm
                x_matrix1 = np.array(mz_list_all[0: final_numbers: number_interval])

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
        error_learning = (np.square(np.array(train_signal) - np.array(y_out_list))).mean()
        print('Test Error:{}'.format(error_learning))
        print('----------------------------------------------------------------')

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


if __name__ == '__main__':
    # ########################################################################################
    # time evolution of resistance
    # ########################################################################################
    # initial state
    a, b, c = 0.1, 0.1, 0
    t_step = 3e-13
    time_consume = 1e-8
    extern_field = 200  # Unit: Oe
    ani_field = 0  # Unit: Oe
    dem_field = 8400  # Unit: Oe
    dc_current = 100
    ac_current = 0
    f_ac = 32e9

    # mtj = Mtj(a, b, c)
    # mtj.stm_train(number_wave=500, nodes_stm=16, visual_process=False, save_index=True, superposition=1,
    #               alert_index=False, time_consume_all=3e-8, ac_amplitude=0)
    # mtj.stm_test(test_number=30, nodes_stm=16, superposition=1, visual_index=True, time_consume_all=3e-8,
    #              ac_amplitude=0)

    # chaos_mine()
    # # ac_current_list = np.linspace(10, 100, 91)
    # ac_current_list = [0, 10, 20, 30, 40, 50]
    # le_list = []
    # for i in range(len(ac_current_list)):
    #     le, _ = edge_of_chaos(ac_current1=ac_current_list[i], len_input_number=number_input_signal,
    #                           len_input_pattern=len_input_signal)
    #     le_list.append(le)
    #     print('*****************************************************************************')
    #     print('ac_list : {}'.format(ac_current_list[:i + 1]))
    #     print('le_list : {}'.format(le_list))
    #     np.save('le_data.npy', le_list)
    #     np.save('ac_data.npy', ac_current_list)
    #     print('*****************************************************************************')
    #
    # print('ac_list : {}'.format(ac_current_list))
    # print('le_ist : {}'.format(le_list))
    # plt.figure()
    # plt.plot(ac_current_list, le_list)
    # plt.show()

    delay_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    node_list = [16, 20, 30, 40, 50]
    for node in node_list:
        for i in delay_list:
            mtj = Mtj(a, b, c)
            mtj.stm_train(number_wave=800, nodes_stm=node, visual_process=False, save_index=True, superposition=i,
                          alert_index=False, time_consume_all=3e-8)
            # mtj.stm_test(test_number=30, nodes_stm=node, superposition=i, visual_index=False)

    # mx_list1, my_list1, mz_list1, t_list1 = mtj.time_evolution(dc_amplitude=dc_current, ac_amplitude=ac_current,
    #                                                            time_consumed=time_consume)
    #
    # # FIGURES
    # plt.figure()
    # plt.plot(mx_list1, c='red', label='mx')
    # plt.legend()
    # plt.ylabel('mx')
    # plt.xlabel('Time:ns')
    # plt.ylim(-1, 1)
    #
    # plt.figure()
    # plt.plot(my_list1, c='purple', label='my')
    # plt.legend()
    # plt.ylabel('my')
    # plt.xlabel('Time:ns')
    # plt.ylim(-1, 1)
    #
    # plt.figure()
    # plt.plot(mz_list1, c='orange', label='mz')
    # plt.legend()
    # plt.ylabel('mz')
    # plt.xlabel('Time: {}s'.format(t_step))
    # plt.ylim(-1, 1)
    #
    # plt.figure('chaos')
    # plt.scatter(my_list1[int(len(my_list1) / 5):], mz_list1[int(len(mz_list1) / 5):], c='red')
    # plt.ylabel('$m_Z$')
    # plt.xlabel('$m_Y$')
    # plt.ylim(-1, 1)
    # plt.xlim(-1, 1)
    #
    # plt.show()
    # #############################################################################################

    # #############################################################################################
    # mtj.classification_train(number_wave=300, nodes_classification=100)
    # mtj.classification_test(test_number=50, nodes_classification=100)

    # ################################################################################
    # finding best values to build esn
    # ################################################################################
    # y_out_list, x_final_matrix, plus_visual_mz, minus_visual_mz = [], [], [], []
    # plus_time, minus_time, time_index = [], [], 0
    # plus_time_switch, minus_time_switch = [], []
    # plus_visual_switch, minus_visual_switch = [], []
    # mtj = Mtj(a, b, c)
    # positive_dc_current = 100
    # negative_dc_current = 200
    # input_signals = np.random.randint(0, 2, 20)
    # # input_signals = [0, 1, 0, 1, 0, 1, 0, 1]
    # mz_behavior = []
    #
    # for j in track(range(len(input_signals))):
    #     if input_signals[j] == 1:
    #         dc_current1 = positive_dc_current
    #         _, _, mz_list, t_list2 = mtj.time_evolution(external_field=200, anisotropy_field=0,
    #                                                          demagnetization_field=8400,
    #                                                          dc_amplitude=dc_current1, ac_amplitude=0,
    #                                                          ac_frequency=32e9, time_consumed=1e-8,
    #                                                          time_step=3e-13)
    #
    #         # _, _, mz_list_osc, t_list_osc = mtj.time_evolution(external_field=200, anisotropy_field=0,
    #         #                                                    demagnetization_field=8400, dc_amplitude=220,
    #         #                                                    ac_amplitude=0, ac_frequency=32e9,
    #         #                                                    time_consumed=7e-10,
    #         #                                                    time_step=3e-13)
    #
    #     else:
    #         dc_current1 = negative_dc_current
    #         mx_list2, _, mz_list, t_list2 = mtj.time_evolution(external_field=200, anisotropy_field=0,
    #                                                                 demagnetization_field=8400,
    #                                                                 dc_amplitude=dc_current1, ac_amplitude=0,
    #                                                                 ac_frequency=32e9, time_consumed=1e-8,
    #                                                                 time_step=3e-13)
    #
    #         # _, _, mz_list_osc, t_list_osc = mtj.time_evolution(external_field=200, anisotropy_field=0,
    #         #                                                    demagnetization_field=8400, dc_amplitude=220,
    #         #                                                    ac_amplitude=0, ac_frequency=32e9,
    #         #                                                    time_consumed=7e-10,
    #         #                                                    time_step=3e-13)
    #     # normalization
    #     mz_list_all = mz_list
    #     # mz_list_all = [i / max(map(abs, mz_list_all)) for i in mz_list_all]
    #     # t_list_osc = [i + t_list2[-1] for i in t_list_osc]
    #     t_list_whole = t_list2
    #
    #     # for figures
    #     if input_signals[j] == 1:
    #         plus_visual_mz = plus_visual_mz + mz_list_all
    #         plus_time = plus_time + list(
    #             np.linspace(time_index, time_index + 3e-13 * len(t_list_whole), len(t_list_whole)))
    #         # plus_time_switch = plus_time_switch + list(
    #         #     np.linspace(time_index, time_index + 3e-13 * len(t_list_whole), len(t_list_whole)))[
    #         #                                       -len(t_list_osc):]
    #         # plus_visual_switch = plus_visual_switch + mz_list_all[-len(mz_list_osc):]
    #     else:
    #         minus_visual_mz = minus_visual_mz + mz_list_all
    #         minus_time = minus_time + list(
    #             np.linspace(time_index, time_index + 3e-13 * len(t_list_whole), len(t_list_whole)))
    #         # minus_time_switch = minus_time_switch + list(
    #         #     np.linspace(time_index, time_index + 3e-13 * len(t_list_whole), len(t_list_whole)))[
    #         #                                         -len(t_list_osc):]
    #         # minus_visual_switch = minus_visual_switch + mz_list_all[-len(mz_list_osc):]
    #
    #     time_index = time_index + t_list_whole[-1]
    #
    # # plt.legend()
    # plt.figure('visual')
    # plt.scatter(minus_time, minus_visual_mz, label='{}'.format(negative_dc_current), s=2, c='red')
    # plt.scatter(plus_time, plus_visual_mz, label='{}'.format(positive_dc_current), s=2, c='blue')
    # # plt.scatter(minus_time_switch, minus_visual_switch, c='green', s=2)
    # # plt.scatter(plus_time_switch, plus_visual_switch, c='green', s=2)
    # plt.legend()
    #
    # # plt.figure('input')
    # # plt.plot(input_signals)
    # plt.show()

    # #############################################################################################
    # dc_current_list = [268, 269, 268, 267, 268, 269, 268, 267, 268, 269, 268, 267, 269, 269, 267, 267,
    #                    269, 269, 267, 267]
    #
    # plt.figure('resistance difference')
    # plt.title('Response for different dc current')
    # plt.xlabel('Time interval')
    # plt.ylabel('resistance difference')
    # plt.ylim(-1, 1)
    # print('dc_list : {}'.format(dc_current_list))
    # resistance_dif_list = []
    # for j in range(len(dc_current_list)):
    #     _, _, mz_list1, t_list1 = mtj.time_evolution(extern_field, ani_field, dem_field,
    #                                                  dc_amplitude=dc_current_list[j], ac_amplitude=ac_current,
    #                                                  ac_frequency=f_ac, time_consumed=time_consume, time_step=t_step)
    #     try:
    #         max_extreme, min_extreme = [], []
    #         for i in range(len(mz_list1)):
    #             if i != 0 and i != len(mz_list1)-1:
    #                 if mz_list1[i] > mz_list1[i-1] and mz_list1[i] > mz_list1[i+1]:
    #                     max_extreme.append(mz_list1[i])
    #                 elif mz_list1[i] < mz_list1[i-1] and mz_list1[i] < mz_list1[i+1]:
    #                     min_extreme.append(mz_list1[i])
    #
    #         length_extreme = min(len(max_extreme), len(min_extreme))
    #         print('length:{}'.format(length_extreme))
    #         # resistance_dif_list = []
    #         for i in range(length_extreme):
    #             resistance_dif_list.append(max_extreme[i])
    #
    #     except Exception as e:
    #         print('error in finding max_extreme or min_extreme: {}'.format(e))
    #         print('max_point :{}'.format(len(max_extreme)))
    #         print('min_point :{}'.format(len(min_extreme)))
    #
    # # plt.figure('resistance difference')
    # temp1 = np.linspace(0, len(resistance_dif_list)-1, len(resistance_dif_list))
    # plt.plot(temp1, resistance_dif_list, c='blue')
    # # plt.scatter(temp1, resistance_dif_list, c='blue')
    # plt.legend()
    # plt.figure('input')
    # plt.plot(dc_current_list)
    # plt.show()
    # #############################################################################################

    # #############################################################################################
    # calculation of lyapunov exponent
    # #############################################################################################
    # le_t_list, le_z_list, le_x_list, le_y_list = [], [], [], []
    # dc_current_list = np.linspace(200, 400, 201)
    # # dc_current_list = [200, 201, 202, 269, 258]
    # for j in track(range(len(dc_current_list))):
    #     eigen_values, le_list = mtj.lyapunov_exponent(calculation_time=500, cal_t_step=3e-13,
    #                                                   dc_amplitude=dc_current_list[j], ac_amplitude=20)
    #     le_t_list.append(le_list[3])
    #     le_z_list.append(le_list[2])
    #     le_x_list.append(le_list[0])
    #     le_y_list.append(le_list[1])
    #
    # plt.figure()
    # plt.title('Maximum Lyapunov exponent')
    # plt.xlabel(r'dc current $t$')
    # plt.ylabel(r'Lyapunov exponent')
    # print('__________________________________')
    # print('le_x :{}'.format(le_x_list))
    # print('le_y :{}'.format(le_y_list))
    # print('le_z :{}'.format(le_z_list))
    # print('le_t :{}'.format(le_t_list))
    # print('__________________________________')
    # plt.plot(le_z_list, c='blue', label='m_z')
    # # plt.plot(le_t_list, c='pink', label='m_t')
    # # plt.plot(le_x_list, c='orange', label='m_x')
    # # plt.plot(le_y_list, c='green', label='m_y')
    # x1 = [0 for i in dc_current_list]
    # plt.plot(x1, ls='--', label='zero line', c='black')
    # plt.legend()
    # plt.show()
