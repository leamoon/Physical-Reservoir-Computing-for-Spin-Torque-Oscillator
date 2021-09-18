import os
import numpy as np
import matplotlib.pyplot as plt


def gram_schmidt(un_o_matrix):
    try:
        # distinguish the shape
        x1, x2, x3, x4 = un_o_matrix[0, :], un_o_matrix[1, :], un_o_matrix[2, :], un_o_matrix[3, :]
        x4_new = x4
        x3_new = x3 - (np.dot(x3, x4_new.T) / np.dot(x4_new, x4_new.T))[0, 0] * x4_new
        x2_new = x2 - (np.dot(x2, x4_new.T) / np.dot(x4_new, x4_new.T))[0, 0] * x4_new - (np.dot(
            x2, x3_new.T) / np.dot(x3_new, x3_new.T))[0, 0] * x3_new
        x1_new = x1 - (np.dot(x1, x4_new.T) / np.dot(x4_new, x4_new.T))[0, 0] * x4_new - (np.dot(
            x1, x3_new.T) / np.dot(x3_new, x3_new.T))[0, 0] * x3_new - (
                         np.dot(x1, x2_new.T) / np.dot(x2_new, x2_new.T))[0, 0] * x2_new
        o_matrix = np.vstack([x1_new, x2_new, x3_new, x4_new])

        # normalization
        x1_new = x1_new / np.linalg.norm(x1_new)
        x2_new = x2_new / np.linalg.norm(x2_new)
        x3_new = x3_new / np.linalg.norm(x3_new)
        x4_new = x4_new / np.linalg.norm(x4_new)
        o_matrix_normal = np.vstack([x1_new, x2_new, x3_new, x4_new])

        return o_matrix, o_matrix_normal

    except Exception as e:
        print(e)
        return 0


def waveform_generator(number_wave):
    """
    a function used to create random wave(sine or square)
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
            sine_points = [259.50, 259.52, 259.53, 259.52, 259.50,
                           259.48, 259.47, 259.48]
            wave_points = wave_points + sine_points
        elif i == 1:
            # square function
            square_points = [259.53] * 4 + [259.47] * 4
            wave_points = wave_points + square_points

    print('wave:{}'.format(wave_points))

    return wave_points, list(random_pulse)


class Mtj:
    def __init__(self, x0=0.1, y0=0.1, z0=1):
        # initial states of magnetization
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

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
        mode = pow(self.x0, 2) + pow(self.y0, 2) + pow(self.z0, 2)
        self.x0 = self.x0 / np.sqrt(mode)
        self.y0 = self.y0 / np.sqrt(mode)
        self.z0 = self.z0 / np.sqrt(mode)
        self.m = np.array([self.x0, self.y0, self.z0])

    def parameters_calculation(self, external_field, anisotropy_field, demagnetization_field, dc_amplitude, time_ac,
                               ac_amplitude, ac_frequency):
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

    def step_evolution(self):
        # normalization
        self.m = np.array([self.x0, self.y0, self.z0])
        last_magnetization = self.m

        # K1 calculation Fourth Runge Kutta Method
        x_axis = np.array([1, 0, 0])
        self.stt_amplitude = self.dc_amplitude + self.ac_amplitude * np.cos(self.time_ac * self.ac_frequency)
        self.stt_amplitude = self.stt_amplitude * self.gyo_ratio

        delta_m = -self.gyo_ratio * np.cross(self.m, self.effective_field) + self.damping_factor * (
                -self.gyo_ratio * np.dot(self.m,
                                         self.effective_field) * self.m + self.gyo_ratio * self.effective_field +
                self.stt_amplitude * np.cross(self.m, (self.m * self.x0 - x_axis))) + self.stt_amplitude * np.cross(
            self.m, np.cross(self.m, x_axis))

        delta1_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))

        # K2
        self.m = last_magnetization + 1 / 2 * delta1_m_reduce * t_step
        time_temp = self.time_ac + 1 / 2 * t_step
        self.stt_amplitude = self.dc_amplitude + self.ac_amplitude * np.cos(time_temp * self.ac_frequency)
        self.stt_amplitude = self.stt_amplitude * self.gyo_ratio

        delta_m = -self.gyo_ratio * np.cross(self.m, self.effective_field) + self.damping_factor * (
                -self.gyo_ratio * np.dot(self.m, self.effective_field) * self.m + self.gyo_ratio *
                self.effective_field +
                self.stt_amplitude * np.cross(self.m, (self.m * self.x0 - x_axis))) + self.stt_amplitude * np.cross(
            self.m, np.cross(self.m, x_axis))

        delta2_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))

        # K3
        self.m = last_magnetization + 1 / 2 * delta2_m_reduce * t_step
        time_temp = self.time_ac + 1 / 2 * t_step
        self.stt_amplitude = self.dc_amplitude + self.ac_amplitude * np.cos(time_temp * self.ac_frequency)
        self.stt_amplitude = self.stt_amplitude * self.gyo_ratio

        delta_m = -self.gyo_ratio * np.cross(self.m, self.effective_field) + self.damping_factor * (
                -self.gyo_ratio * np.dot(self.m,
                                         self.effective_field) * self.m + self.gyo_ratio * self.effective_field +
                self.stt_amplitude * np.cross(self.m, (self.m * self.x0 - x_axis))) + self.stt_amplitude * np.cross(
            self.m, np.cross(self.m, x_axis))

        delta3_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))

        # K4
        self.m = last_magnetization + delta3_m_reduce * t_step
        time_temp = self.time_ac + t_step
        self.stt_amplitude = self.dc_amplitude + self.ac_amplitude * np.cos(time_temp * self.ac_frequency)
        self.stt_amplitude = self.stt_amplitude * self.gyo_ratio

        delta_m = -self.gyo_ratio * np.cross(self.m, self.effective_field) + self.damping_factor * (
                -self.gyo_ratio * np.dot(self.m,
                                         self.effective_field) * self.m + self.gyo_ratio * self.effective_field +
                self.stt_amplitude * np.cross(self.m, (self.m * self.x0 - x_axis))) + self.stt_amplitude * np.cross(
            self.m, np.cross(self.m, x_axis))
        delta4_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))

        self.m = last_magnetization + 1 / 6 * t_step * (
                delta1_m_reduce + 2 * delta2_m_reduce + 2 * delta3_m_reduce + delta4_m_reduce)

        # normalization of magnetization vector
        [self.x0, self.y0, self.z0] = self.m
        mode = pow(self.x0, 2) + pow(self.y0, 2) + pow(self.z0, 2)
        self.x0 = self.x0 / np.sqrt(mode)
        self.y0 = self.y0 / np.sqrt(mode)
        self.z0 = self.z0 / np.sqrt(mode)
        self.m = np.array([self.x0, self.y0, self.z0])

        # recover
        self.stt_amplitude = self.dc_amplitude + self.ac_amplitude * np.cos(self.time_ac * self.ac_frequency)
        return self.x0, self.y0, self.z0

    def frequency_dc(self, external_field, anisotropy_field, demagnetization_field, dc_amplitude,
                     ac_amplitude, ac_frequency, time_consumed, time_step):
        mx_list, my_list, mz_list, t_list = [], [], [], []

        # time evolution
        for i1 in range(int(time_consumed / time_step)):
            self.parameters_calculation(external_field, anisotropy_field, demagnetization_field, dc_amplitude,
                                        i1 * time_step,
                                        ac_amplitude, ac_frequency)
            self.step_evolution()
            mx_list.append(self.x0)
            my_list.append(self.y0)
            mz_list.append(self.z0)
            t_list.append(i1 * time_step)

            # calculation of time
            if i1 % 1000 == 0:
                print('Process: {:.3} %  Time evolution'.format((i1 + 1) / int(time_consumed / time_step) * 100))

        # find dc frequency
        extreme_points = []
        for i1 in range(len(mz_list)):
            if i1 != 0 and i1 != len(mz_list) - 1:
                if mz_list[i1] < mz_list[i1 - 1] and mz_list[i1] < mz_list[i1 + 1]:
                    extreme_points.append(t_list[i1])
        self.dc_frequency = 1 / (extreme_points[int(len(extreme_points) / 2)] - extreme_points[int(len(
            extreme_points) / 2) - 1])
        # print('dc frequency: {}'.format(self.dc_frequency))

        return mx_list, my_list, mz_list, t_list

    def lyapunov_parameter(self, current_magnetization, current_t, time_interval, delta_x=None):
        """
        a function to get largest lyapunov exponent and its spectrum
        :param current_t: the current time, which work for stt term
        :param time_interval: the time step between two closest elements
        :param delta_x: the input orthogonal matrix
        :param current_magnetization: the magnetization states at current time
        :return v_matrix, delta_x
        """

        # define the g parameters, to simplified latter calculation
        [self.x0, self.y0, self.z0] = current_magnetization
        self.stt_amplitude = self.dc_amplitude + self.ac_amplitude*np.cos(self.ac_frequency*current_t)
        stt_partial = -self.ac_amplitude*self.ac_frequency*np.sin(self.ac_frequency*current_t)

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
                        self.damping_factor*(pow(self.y0, 2)+pow(self.z0, 2))*stt_partial) / (
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
                                                                self.demagnetization_field*2*self.z0 * g3 * self.y0) -
                        self.damping_factor * self.gyo_ratio * self.stt_amplitude * g3 -
                        self.damping_factor*self.gyo_ratio*stt_partial*self.z0 +
                        self.damping_factor * self.stt_amplitude * (self.x0 * g2 + self.y0 * g1) +
                        self.damping_factor*stt_partial*self.x0*self.y0) / (1 + self.damping_factor ** 2)

        partial_g3_mx = (self.gyo_ratio * self.y0 * self.anisotropy_field -
                         self.damping_factor * self.gyo_ratio * (
                                 self.external_field + self.x0 * 2 * self.anisotropy_field) * self.z0 +
                         self.damping_factor * self.stt_amplitude * self.z0
                         ) / (1 + self.damping_factor ** 2)

        partial_g3_my = self.gyo_ratio * (self.external_field + self.anisotropy_field * self.x0 +
                                          self.stt_amplitude * self.damping_factor) / (1 + self.damping_factor ** 2)

        partial_g3_mz = (-self.damping_factor * self.gyo_ratio * (self.x0 * self.external_field +
                                                                  self.anisotropy_field * pow(self.x0, 2) +
                                                                  self.demagnetization_field * pow(self.z0, 3)*3) +
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
                        self.damping_factor*self.gyo_ratio*stt_partial*self.y0 +
                        self.stt_amplitude * self.damping_factor * (g1 * self.z0 + g3 * self.x0) +
                        self.damping_factor*self.x0*self.z0*stt_partial) / (
                                   1 + self.damping_factor ** 2)

        # initial orthogonal matrix
        if delta_x is None:
            delta_x = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        jacobian_matrix = np.mat([[partial_g1_mx, partial_g1_my, partial_g1_mz, partial_g1_t],
                                  [partial_g2_mx, partial_g2_my, partial_g2_mz, partial_g2_t],
                                  [partial_g3_mx, partial_g3_my, partial_g3_mz, partial_g3_t],
                                  [0, 0, 0, 0]])

        delta_x_interval = np.dot(jacobian_matrix, delta_x) * time_interval
        delta_x_new = delta_x + delta_x_interval
        # Gram-Schmidt Orthogonalization
        v_matrix, delta_x = gram_schmidt(delta_x_new)
        return v_matrix, delta_x

    def lyapunov_exponent(self, current_magnetization, current_time, index_time, length_n=10, cal_t_step=3e-13):

        delta_x = None
        sum_x, sum_y, sum_z, sum_t = 0, 0, 0, 0
        [mx, my, mz] = current_magnetization
        for i in range(int(length_n)):
            # find corresponding magnetization to the current_t
            v_matrix, delta_x = self.lyapunov_parameter(current_t=cal_t_step*i+current_time, time_interval=cal_t_step,
                                                        delta_x=delta_x, current_magnetization=[mx[index_time+i],
                                                                                                my[index_time+i],
                                                                                                mz[index_time+i]])
            # print('delta_x:{}'.format(delta_x))
            sum_x = np.log(np.linalg.norm(v_matrix[0, :])) + sum_x
            sum_y = np.log(np.linalg.norm(v_matrix[1, :])) + sum_y
            sum_z = np.log(np.linalg.norm(v_matrix[2, :])) + sum_z
            sum_t = np.log(np.linalg.norm(v_matrix[3, :])) + sum_t

        # calculation of Le
        sum_x = sum_x / length_n
        sum_y = sum_y / length_n
        sum_z = sum_z / length_n / cal_t_step * 1e-9
        sum_t = sum_t / length_n

        return sum_x, sum_y, sum_z, sum_t

    def classification_train(self, number_wave, nodes_classification):
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
        y_out_list, x_final_matrix = [], []
        s_in, train_signal = waveform_generator(number_wave)
        print('---------------------------------------------------------------')

        for i in range(len(s_in)):
            mx_list2, my_list2, mz_list2, t_list2 = self.frequency_dc(external_field=200, anisotropy_field=0,
                                                                      demagnetization_field=8400, dc_amplitude=s_in[i],
                                                                      ac_amplitude=20, ac_frequency=32e9,
                                                                      time_consumed=1e-8, time_step=3e-13)

            # sampling points
            number_interval = int(len(mz_list2) / nodes_classification)
            if number_interval < 1:
                number_interval = 1
            x_matrix1 = np.array(mz_list2[1: len(mz_list2):number_interval])

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
                print('progress : {:.3} % classification training'.format(i/len(s_in)*100))
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

    def classification_test(self, test_number, nodes_classification):
        if os.path.exists('weight_matrix_chaos/weight_out_classification.npy'):
            weight_out_stm = np.load('weight_matrix_chaos/weight_out_classification.npy')
            print('\r' + 'Loading weight_out_classification matrix successfully !', end='', flush=True)

        else:
            print('\rno valid weight data !', end='', flush=True)
            return 0

        print('\rClassification')
        print('----------------------------------------------------------------')
        print('start to test !', flush=True)

        # fabricate the input, target, and output signals
        y_out_list, x_final_matrix = [], []
        s_in, train_signal = waveform_generator(test_number)
        print('---------------------------------------------------------------')

        for i in s_in:
            mx_list2, my_list2, mz_list2, t_list2 = self.frequency_dc(external_field=200, anisotropy_field=0,
                                                                      demagnetization_field=8400, dc_amplitude=i,
                                                                      ac_amplitude=20, ac_frequency=32e9,
                                                                      time_consumed=1e-7, time_step=3e-13)

            # sampling points
            number_interval = int(len(mz_list2) / nodes_classification)
            print('length of mz_list form: {}'.format(len(mz_list2)))
            if number_interval < 1:
                number_interval = 1
            x_matrix1 = np.array(mz_list2[1: len(mz_list2):number_interval])

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

        # calculate the error
        error_learning = np.var(np.array(train_signal) - np.array(y_out_list))
        print('test error:{}'.format(error_learning))
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


if __name__ == '__main__':
    # initial state
    a, b, c = 0.1, 0.1, 0
    t_step = 3e-13
    time_consume = 1e-8
    number_intervals = int(time_consume / t_step)
    extern_field = 200  # Unit: Oe
    ani_field = 0  # Unit: Oe
    dem_field = 8400  # Unit: Oe
    dc_current = 265
    ac_current = 20
    f_ac = 32e9

    mtj = Mtj(a, b, c)
    # mtj.lyapunov_exponent(length_cal=10, time_interval=1e-9)

    mx_list1, my_list1, mz_list1, t_list1 = mtj.frequency_dc(extern_field, ani_field, dem_field, dc_current,
                                                             ac_amplitude=ac_current, ac_frequency=f_ac,
                                                             time_consumed=time_consume, time_step=t_step)

    temp1 = mz_list1[200:]
    fourier_consequence = np.abs(np.fft.fft(temp1))
    fourier_consequence = fourier_consequence / max(fourier_consequence) * 50
    fourier_freq = np.fft.fftfreq(len(t_list1))
    pds = [pow(i, 2) for i in fourier_consequence]
    # print('Fourier:{}'.format(pds))

    # #############################################################################################
    # calculation of lyapunov exponent
    # #############################################################################################
    # le_t_list, le_z_list, le_x_list, le_y_list = [], [], [], []
    # length_cal = 1000
    # for i in range(len(t_list1)-length_cal):
    #     le_x, le_y, le_z, le_t = mtj.lyapunov_exponent(current_time=t_list1[i], length_n=10,
    #                                                    cal_t_step=t_step, index_time=int(i),
    #                                                    current_magnetization=[mx_list1,
    #                                                                           my_list1,
    #                                                                           mz_list1])
    #     le_t_list.append(le_t)
    #     le_z_list.append(le_z)
    #     le_x_list.append(le_x)
    #     le_y_list.append(le_y)
    #
    #     # time index
    #     if i % 1000 == 0:
    #         print('Process: {:.3} %  Calculating Lyapunov Exponent'.format((i + 1) / len(t_list1) * 100))
    #
    # print('max :{}'.format(np.max(le_z_list)))
    # plt.figure()
    # plt.title('Lyapunov exponent')
    # plt.xlabel(r'Time $t$')
    # plt.ylabel(r'Lyapunov exponent')
    # plt.plot(t_list1[:-int(length_cal)], le_z_list, c='blue', label='m_z')
    # plt.plot(t_list1[:-int(length_cal)], le_t_list, c='pink', label='m_t')
    # # plt.plot(t_list1[:-int(length_cal)], le_x_list, c='orange', label='m_x')
    # # plt.plot(t_list1[:-int(length_cal)], le_y_list, c='green', label='m_y')
    # time_consequence = [0 for i in t_list1]
    # plt.plot(t_list1, time_consequence, ls='--', label='zero line', c='black')
    # plt.legend()
    # # plt.ylim(-0.1, 0.5)

    plt.figure()
    plt.plot(mx_list1, c='red', label='mx')
    plt.legend()
    plt.ylabel('mx')
    plt.xlabel('Time:ns')
    plt.ylim(-1, 1)

    plt.figure()
    plt.plot(my_list1, c='purple', label='my')
    plt.legend()
    plt.ylabel('my')
    plt.xlabel('Time:ns')
    plt.ylim(-1, 1)

    plt.figure()
    plt.plot(mz_list1, c='orange', label='mz')
    plt.legend()
    plt.ylabel('mz')
    plt.xlabel('Time: {}s'.format(t_step))
    plt.ylim(-1, 1)

    plt.figure('chaos')
    plt.scatter(my_list1[int(len(my_list1) / 5):], mz_list1[int(len(mz_list1) / 5):], c='red')
    plt.ylabel('$m_Z$')
    plt.xlabel('$m_Y$')
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)

    plt.figure('PDS')
    fre_x_list = np.linspace(-1, 1, len(pds) - 1)
    plt.plot(fre_x_list, pds[1:])
    plt.ylabel('Fourier Transform')
    plt.xlabel('Frequency interval')

    plt.show()
