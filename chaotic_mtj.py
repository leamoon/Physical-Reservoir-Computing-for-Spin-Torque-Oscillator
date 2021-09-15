import numpy as np
import matplotlib.pyplot as plt


def gram_schmidt(un_o_matrix):
    try:
        # distinguish the shape
        row_len, column_len = un_o_matrix.shape
        x1, x2, x3, x4 = un_o_matrix[0, :], un_o_matrix[1, :], un_o_matrix[2, :], un_o_matrix[3, :]
        x1_new = x1
        x2_new = x2 - (np.dot(x2, x1_new.T) / np.dot(x1_new, x1_new.T))[0, 0] * x1_new
        x3_new = x3 - (np.dot(x3, x1_new.T) / np.dot(x1_new, x1_new.T))[0, 0] * x1_new - (np.dot(
            x3, x2_new.T) / np.dot(x2_new, x2_new.T))[0, 0] * x2_new
        x4_new = x4 - (np.dot(x4, x1_new.T) / np.dot(x1_new, x1_new.T))[0, 0] * x1_new - (np.dot(
            x4, x2_new.T) / np.dot(x2_new, x2_new.T))[0, 0] * x2_new - (
                         np.dot(x4, x3_new.T) / np.dot(x3_new, x3_new.T))[0, 0] * x3_new
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
                print('Process: {:.3} %'.format((i1 + 1) / int(time_consumed / time_step) * 100))

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

    def lyapunov_parameter(self, current_t, time_interval, delta_x=None):
        """
        a function to get largest lyapunov exponent and its spectrum
        :param current_t: the current time, which work for stt term
        :param time_interval: the time step between two closest elements
        :param delta_x: the input orthogonal matrix
        :return v_matrix, delta_x
        """

        # define the g parameters, to simplified latter calculation
        self.stt_amplitude = self.dc_amplitude + self.ac_amplitude*np.cos(self.ac_frequency*current_t)
        stt_partial = self.ac_frequency*self.ac_frequency*np.cos(self.ac_frequency*current_t)

        g1 = (-self.gyo_ratio * self.z0 * self.y0 * self.demagnetization_field -
              self.damping_factor * self.gyo_ratio * (self.x0 * self.external_field +
                                                      pow(self.x0, 2) * self.anisotropy_field +
                                                      self.demagnetization_field * pow(self.z0, 2)) * self.x0 +
              self.damping_factor * self.gyo_ratio * (self.external_field + self.anisotropy_field * self.x0) +
              self.damping_factor * self.stt_amplitude * self.gyo_ratio * self.x0 -
              (pow(self.y0, 2) + pow(self.z0, 2)) * self.stt_amplitude * self.damping_factor) / (
                         1 + self.damping_factor ** 2)

        g2 = (-self.gyo_ratio * (-self.x0 * self.z0 * self.demagnetization_field +
                                 self.z0 * self.external_field + self.z0 * self.x0 * self.anisotropy_field) -
              self.damping_factor * self.gyo_ratio * (self.x0 * self.external_field +
                                                      self.anisotropy_field * pow(self.x0, 2) +
                                                      self.demagnetization_field * pow(self.z0, 2)) * self.y0 +
              self.damping_factor * self.gyo_ratio * self.stt_amplitude * (self.x0 - self.z0) +
              self.x0 * self.y0 * self.stt_amplitude * self.damping_factor) / (1 + self.damping_factor ** 2)

        g3 = (self.gyo_ratio * (self.y0 * self.external_field + self.x0 * self.y0 * self.anisotropy_field) -
              self.damping_factor * self.gyo_ratio * (self.x0 * self.external_field +
                                                      self.anisotropy_field * pow(self.x0, 2) +
                                                      self.demagnetization_field * pow(self.z0, 2)) * self.z0 +
              self.damping_factor * self.gyo_ratio * (self.demagnetization_field * self.z0 +
                                                      self.x0 * self.stt_amplitude +
                                                      self.stt_amplitude * self.y0) +
              self.stt_amplitude * self.damping_factor * self.x0 * self.z0) / (1 + self.damping_factor ** 2)

        # construct jacobian matrix
        partial_g1_mx = -self.damping_factor * self.gyo_ratio * (
                2 * self.x0 * self.external_field + 3 * pow(self.x0, 2) * self.anisotropy_field +
                self.demagnetization_field * pow(self.z0, 2) - self.anisotropy_field - self.stt_amplitude) / (
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
                        self.damping_factor * self.gyo_ratio * self.anisotropy_field * g1 +
                        self.damping_factor*self.gyo_ratio*(stt_partial*self.x0 + self.stt_amplitude*g1) -
                        self.stt_amplitude * self.damping_factor * (2 * self.y0 * g2 + 2 * self.z0 * g3) +
                        self.damping_factor*(pow(self.y0, 2)+pow(self.z0, 2))*stt_partial) / (
                                   1 + self.damping_factor ** 2)

        partial_g2_mx = (-self.gyo_ratio * (-self.z0 * self.demagnetization_field + self.z0 * self.anisotropy_field) -
                         self.damping_factor * self.gyo_ratio * (self.y0 * self.external_field +
                                                                 2 * self.x0 * self.y0 * self.anisotropy_field) +
                         self.damping_factor * self.gyo_ratio * self.stt_amplitude +
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
                                                                self.demagnetization_field*2*self.z0 * g3 * self.y0) +
                        self.damping_factor * self.gyo_ratio * self.stt_amplitude * (g1 - g3) +
                        self.damping_factor*self.gyo_ratio*(stt_partial*self.x0-stt_partial*self.z0) +
                        self.damping_factor * self.stt_amplitude * (self.x0 * g2 + self.y0 * g1) +
                        self.damping_factor*stt_partial*self.x0*self.y0) / (1 + self.damping_factor ** 2)

        partial_g3_mx = (self.gyo_ratio * self.y0 * self.anisotropy_field -
                         self.damping_factor * self.gyo_ratio * (
                                 self.external_field + self.x0 * 2 * self.anisotropy_field) * self.z0 +
                         self.damping_factor * self.gyo_ratio * self.stt_amplitude +
                         self.damping_factor * self.stt_amplitude * self.z0
                         ) / (1 + self.damping_factor ** 2)

        partial_g3_my = self.gyo_ratio * (self.external_field + self.anisotropy_field * self.x0 +
                                          self.stt_amplitude * self.damping_factor) / (1 + self.damping_factor ** 2)

        partial_g3_mz = (-self.damping_factor * self.gyo_ratio * (self.x0 * self.external_field +
                                                                  self.anisotropy_field * pow(self.x0, 2) +
                                                                  self.demagnetization_field * pow(self.z0, 3)) +
                         self.damping_factor * self.gyo_ratio * self.demagnetization_field +
                         self.stt_amplitude * self.damping_factor * self.x0) / (1 + self.damping_factor ** 2)

        partial_g3_t = (self.gyo_ratio * (
                    g2 * self.external_field + self.anisotropy_field * (self.x0 * g2 + g1 * self.y0)) -
                        self.damping_factor * self.gyo_ratio * (g1 * self.z0 * self.external_field +
                                                                g2 * self.x0 * self.external_field +
                                                                self.anisotropy_field * g3 * pow(self.x0, 2) +
                                                                self.anisotropy_field * 2 * g1 * self.z0 * self.x0 +
                                                                self.demagnetization_field * g3 * 3 * pow(self.z0, 2)) +
                        self.damping_factor * self.gyo_ratio * (self.demagnetization_field * g3 +
                                                                self.stt_amplitude * g1 +
                                                                self.stt_amplitude * g2) +
                        self.damping_factor*self.gyo_ratio*(stt_partial*self.x0+stt_partial*self.y0) +
                        self.stt_amplitude * self.damping_factor * (g1 * self.z0 + g3 * self.x0) +
                        self.damping_factor*self.x0*self.z0*stt_partial) / (
                                   1 + self.damping_factor ** 2)

        # initial orthogonal matrix
        if delta_x is None:
            delta_x = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            print('Create the initial delta_x')
        jacobian_matrix = np.mat([[partial_g1_mx, partial_g1_my, partial_g1_mz, partial_g1_t],
                                  [partial_g2_mx, partial_g2_my, partial_g2_mz, partial_g2_t],
                                  [partial_g3_mx, partial_g3_my, partial_g3_mz, partial_g3_t],
                                  [0, 0, 0, 0]])

        delta_x_interval = np.dot(jacobian_matrix, delta_x) * time_interval
        delta_x_new = delta_x + delta_x_interval
        # Gram-Schmidt Orthogonalization
        v_matrix, delta_x = gram_schmidt(delta_x_new)
        return v_matrix, delta_x

    def lyapunov_exponent(self, current_time, length_n=10, cal_t_step=0.01):

        delta_x = None
        sum_x, sum_y, sum_z, sum_t = 0, 0, 0, 0
        for i in range(int(length_n)):
            v_matrix, delta_x = self.lyapunov_parameter(current_t=cal_t_step*i+current_time, time_interval=cal_t_step,
                                                        delta_x=delta_x)
            sum_x = np.log(np.linalg.norm(v_matrix[0, :])) + sum_x
            sum_y = np.log(np.linalg.norm(v_matrix[1, :])) + sum_y
            sum_z = np.log(np.linalg.norm(v_matrix[2, :])) + sum_z
            sum_t = np.log(np.linalg.norm(v_matrix[3, :])) + sum_t

        # calculation of Le
        sum_x = sum_x / length_n / cal_t_step
        sum_y = sum_y / length_n / cal_t_step
        sum_z = sum_z / length_n / cal_t_step
        sum_t = sum_t / length_n / cal_t_step

        return sum_x, sum_y, sum_z, sum_t


if __name__ == '__main__':
    # initial state
    a, b, c = 0.1, 0.1, 0
    t_step = 3e-13
    time_consume = 1e-8
    number_intervals = int(time_consume / t_step)
    extern_field = 200  # Unit: Oe
    ani_field = 0  # Unit: Oe
    dem_field = 8400  # Unit: Oe
    dc_current = 420
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

    mz_list1 = np.array(mz_list1).reshape(-1, 1)
    le_x, le_y, le_z, le_t = mtj.lyapunov_exponent(current_time=100e-9, length_n=100, cal_t_step=1e-11)
    print('le_z: {}'.format(le_z))
    print('le_t: {}'.format(le_t))
    temp1_list = np.linspace(0, 100-9, 101)*1e-9
    le_t_list, le_z_list, le_x_list, le_y_list = [], [], [], []
    for i in temp1_list:
        le_x, le_y, le_z, le_t = mtj.lyapunov_exponent(current_time=i, length_n=50, cal_t_step=1e-11)
        le_t_list.append(le_t)
        le_z_list.append(le_z)
        le_x_list.append(le_x)
        le_y_list.append(le_y)

    plt.figure()
    plt.title('Maximum Lyapunov exponent')
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'Maximum Lyapunov exponent')
    plt.plot(temp1_list, le_z_list, c='blue', label='m_z')
    plt.plot(temp1_list, le_t_list, c='pink', label='m_t')
    plt.plot(temp1_list, le_x_list, c='orange', label='m_x')
    plt.plot(temp1_list, le_y_list, c='green', label='m_y')
    x1 = [0 for i in temp1_list]
    plt.plot(temp1_list, x1, label='zero line')
    plt.legend()

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
