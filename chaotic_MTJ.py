import numpy as np
import matplotlib.pyplot as plt
from nolitsa import lyapunov


class Mtj:
    def __init__(self, x0, y0, z0):
        # initial states of magnetization
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.m = np.array([self.x0, self.y0, self.z0])

        # hyper parameters
        self.damping_factor = 0.02
        self.gyo_ratio = 1.7e7  # unit Oe^-1 s^-1

        # declare variables
        self.external_field, self.anisotropy_field, self.demagnetization_field, self.dl_field = 0, 0, 0, 0
        self.effective_field = 0
        self.stt_amplitude = 0
        self.tao_1, self.tao_2, self.tao_3, self.tao_4 = None, None, None, None
        self.dc_frequency = 0

        self.theta, self.phi = None, None

        # normalization of magnetization vector
        mode = pow(self.x0, 2) + pow(self.y0, 2) + pow(self.z0, 2)
        self.x0 = self.x0 / np.sqrt(mode)
        self.y0 = self.y0 / np.sqrt(mode)
        self.z0 = self.z0 / np.sqrt(mode)

    def parameters_calculation(self, external_field, anisotropy_field, demagnetization_field, dc_amplitude, time_ac,
                               ac_amplitude, ac_frequency):
        # effective field
        self.external_field = external_field
        self.anisotropy_field = anisotropy_field
        self.demagnetization_field = demagnetization_field
        self.effective_field = np.array([self.external_field + self.anisotropy_field * self.x0, 0,
                                         -self.demagnetization_field * self.z0])

        # STT term
        self.stt_amplitude = dc_amplitude + ac_amplitude * np.cos(time_ac * ac_frequency)

        # damping like field
        self.dl_field = - self.damping_factor / self.x0 * (
                self.external_field * self.x0 + self.anisotropy_field + self.demagnetization_field / 2)

    def step_evolution(self):
        # time evolution
        self.m = np.array([self.x0, self.y0, self.z0])
        last_magnetization = self.m

        # K1 calculation Fourth Runge Kutta Method
        x_axis = np.array([1, 0, 0])

        delta_m = -self.gyo_ratio * np.cross(self.m, self.effective_field) - self.damping_factor * self.gyo_ratio * \
                  np.dot(
                      self.m,
                      self.effective_field) * self.m + self.damping_factor * self.gyo_ratio * self.effective_field + self.stt_amplitude * self.damping_factor * self.gyo_ratio * (
                          self.x0 * np.cross(self.m, self.m) - np.cross(self.m, x_axis)) + (self.m * np.dot(self.m, x_axis) - x_axis) * self.stt_amplitude * self.gyo_ratio

        delta1_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))

        # K2
        self.m = last_magnetization + 1 / 2 * delta1_m_reduce * t_step
        delta_m = -self.gyo_ratio * np.cross(self.m,
                                             self.effective_field) - self.damping_factor * self.gyo_ratio * np.dot(
            self.m, self.effective_field) * self.m + self.damping_factor * self.gyo_ratio * self.effective_field + self \
                      .stt_amplitude * self.damping_factor * self.gyo_ratio * (
                          self.x0 * np.cross(self.m, self.m) - np.cross(self.m, x_axis)) + \
                  (self.m * np.dot(self.m, x_axis) - x_axis) * self.stt_amplitude * self.gyo_ratio

        delta2_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))

        # K3
        self.m = last_magnetization + 1 / 2 * delta2_m_reduce * t_step
        delta_m = -self.gyo_ratio * np.cross(self.m,
                                             self.effective_field) - self.damping_factor * self.gyo_ratio * np.dot(
            self.m, self.effective_field) * self.m + self.damping_factor * self.gyo_ratio * self.effective_field + self \
                      .stt_amplitude * self.damping_factor * self.gyo_ratio * (
                          self.x0 * np.cross(self.m, self.m) - np.cross(self.m, x_axis)) + \
                  (self.m * np.dot(self.m, x_axis) - x_axis) * self.stt_amplitude * self.gyo_ratio

        delta3_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))

        # K4
        self.m = last_magnetization + delta3_m_reduce * t_step
        delta_m = -self.gyo_ratio * np.cross(self.m,
                                             self.effective_field) - self.damping_factor * self.gyo_ratio * np.dot(
            self.m, self.effective_field) * self.m + self.damping_factor * self.gyo_ratio * self.effective_field + self \
                      .stt_amplitude * self.damping_factor * self.gyo_ratio * (
                          self.x0 * np.cross(self.m, self.m) - np.cross(self.m, x_axis)) + \
                  (self.m * np.dot(self.m, x_axis) - x_axis) * self.stt_amplitude * self.gyo_ratio
        delta4_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))

        self.m = last_magnetization + 1 / 6 * t_step * (
                    delta1_m_reduce + 2 * delta2_m_reduce + 2 * delta3_m_reduce + delta4_m_reduce)

        # normalization of magnetization vector
        [self.x0, self.y0, self.z0] = self.m
        mode = pow(self.x0, 2) + pow(self.y0, 2) + pow(self.z0, 2)
        self.x0 = self.x0 / np.sqrt(mode)
        self.y0 = self.y0 / np.sqrt(mode)
        self.z0 = self.z0 / np.sqrt(mode)
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

        # find dc frequency
        extreme_points = []
        for i1 in range(len(mz_list)):
            if i1 != 0 and i1 != len(mz_list) - 1:
                if mz_list[i1] < mz_list[i1 - 1] and mz_list[i1] < mz_list[i1 + 1]:
                    extreme_points.append(t_list[i1])
        self.dc_frequency = 1 / (extreme_points[int(len(extreme_points) / 2)] - extreme_points[int(len(
            extreme_points) / 2) - 1])
        print('dc frequency: {}'.format(self.dc_frequency))

        return mx_list, my_list, mz_list, t_list

    # def lyapunov_exponent(self, length_cal, time_interval):
    # self.theta = np.arccos(self.x0)
    # self.phi = np.arctan(self.y0/self.z0)
    #
    # partial_11 = (-self.gyo_ratio*self.demagnetization_field*np.sin(self.phi)*np.cos(self.phi)*np.cos(self.theta) -
    #               self.damping_factor*self.gyo_ratio*self.external_field*np.cos(
    #             self.theta)-self.damping_factor*self.gyo_ratio*self.anisotropy_field*np.cos(self.theta)*np.cos(
    #             self.theta)+self.damping_factor*self.gyo_ratio*self.anisotropy_field*np.sin(self.theta)*np.sin(
    #             self.theta)-self.damping_factor*self.gyo_ratio*self.demagnetization_field*np.cos(
    #             2*self.theta)*np.cos(self.phi)*np.cos(self.phi)-self.gyo_ratio*self.dl_field*np.cos(
    #             self.theta))/(1+self.damping_factor**2)
    #
    # partial_12 = (-self.gyo_ratio*self.demagnetization_field*np.sin(self.theta)*np.cos(
    #     2*self.phi)+self.gyo_ratio*self.damping_factor*self.demagnetization_field*np.sin(self.theta)*np.cos(
    #     self.theta)*np.sin(2*self.phi))/(1+self.damping_factor**2)
    #
    # partial_13 = 0
    #
    # partial_21 = (self.gyo_ratio*self.anisotropy_field*np.sin(
    #     self.theta)+self.gyo_ratio*self.demagnetization_field*pow(np.cos(self.phi), 2))/(
    #         1+self.damping_factor**2)
    #
    # partial_22 = (self.gyo_ratio*self.demagnetization_field*np.cos(self.theta)*np.sin(
    #     self.phi*2)+self.damping_factor*self.gyo_ratio*self.demagnetization_field*np.cos(2*self.phi))/(
    #         1+self.damping_factor**2)
    #
    # partial_23 = 0
    #
    # partial_31, partial_32, partial_33 = 0, 0, 0
    #
    # # build a jacobian matrix
    # jacobian_matrix = np.matrix([[partial_11, partial_12, partial_13], [partial_21, partial_22, partial_23],
    #                              [partial_31, partial_32, partial_33]])
    #
    # initial_orthogonal_vec = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # v_vector_list = []
    # # try:
    # #     for i in range(int(length_cal)):
    # #         temp1 = np.dot(jacobian_matrix, initial_orthogonal_vec)
    # #         temp2 = np.multiply(temp1, time_interval) + initial_orthogonal_vec
    # #         # orthogonalization
    # #         temp2 = [sympy.Matrix(i) for i in temp2]
    # #         v_vector = sympy.GramSchmidt(temp2)
    # #         temp3 = [np.matrix(i) for i in v_vector]
    # #         v_vector = np.vstack((temp3[0], temp3[1], temp3[2]))
    # #         v_vector_list.append(v_vector)
    # #         print('v_vector:{} for {}'.format(v_vector, i))
    # #         initial_orthogonal_vec = v_vector
    # #         print('Normalization:{}'.format(initial_orthogonal_vec))
    # #     print(v_vector_list)
    # #
    # # except ValueError as e:
    # #     print('------------------------------------------------------------')
    # #     print('-------------    error   -----------------------------------')
    # #     print(e)
    # #     print('------------------------------------------------------------')
    # #     print('------------------------------------------------------------')
    # #     return 0
    #
    # # another way to calculate the Lyapunov Exponent


if __name__ == '__main__':
    # initial state
    a, b, c = 0.1, 0.1, 0
    t_step = 1e-11
    time_consume = 1e-7
    number_intervals = int(time_consume / t_step)
    extern_field = 200  # Unit: Oe
    ani_field = 0  # Unit: Oe
    dem_field = 8400  # Unit: Oe
    dc_current = 259.54
    ac_current = 20
    f_ac = 30e9

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
    print('Fourier:{}'.format(pds))

    mz_list1 = np.array(mz_list1).reshape(-1, 1)
    d = lyapunov.mle(mz_list1[int(len(mz_list1) / 2):], maxt=10, window=1e-7, maxnum=100)

    plt.figure()
    plt.title('Maximum Lyapunov exponent')
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'Maximum Lyapunov exponent')
    plt.plot(d, label='divergence')
    t1 = np.linspace(0, len(d) - 1, len(d))
    x1 = [0 for i in t1]
    plt.plot(t1, x1, label='zero line')
    plt.legend()

    # plt.figure()
    # plt.plot(mx_list1, c='red', label='mx')
    # plt.legend()
    # plt.ylabel('mx')
    # plt.xlabel('Time:ns')
    # plt.ylim(-1, 1)

    # plt.figure()
    # plt.plot(my_list1, c='purple', label='my')
    # plt.legend()
    # plt.ylabel('my')
    # plt.xlabel('Time:ns')
    # plt.ylim(-1, 1)

    plt.figure()
    plt.plot(mz_list1, c='orange', label='mz')
    plt.legend()
    plt.ylabel('mz')
    plt.xlabel('Time:ns')
    plt.ylim(-1, 1)

    plt.figure('chaos')
    plt.scatter(my_list1, mz_list1, c='red')
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

    # ##################################################################################
    # find chaos
    # ###################################################################################
    # le_list = []
    # fac_list = np.linspace(226, 232, 50)
    # for i in fac_list:
    #     mtj = Mtj(a, b, c)
    #
    #     mx_list1, my_list1, mz_list1, t_list1 = mtj.frequency_dc(extern_field, ani_field, dem_field, i,
    #                                                              ac_amplitude=ac_current, ac_frequency=f_ac,
    #                                                              time_consumed=time_consume, time_step=t_step)
    #
    #     mz_list1 = np.array(mz_list1).reshape(-1, 1)
    #     print(mz_list1.shape)
    #     d = lyapunov.mle(mz_list1, maxt=10, window=1e-3, maxnum=10)
    #     le_list.append(max(d))
    #
    # plt.figure()
    # plt.plot(fac_list, le_list, c='blue')
    # plt.ylabel('Lyapunov Exponent')
    # plt.xlabel('dc current(Oe)')
    # plt.show()
