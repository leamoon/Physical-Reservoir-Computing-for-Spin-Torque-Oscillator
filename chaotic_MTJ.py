import numpy
import numpy as np
import matplotlib.pyplot as plt


class Mtj:
    def __init__(self, x0, y0, z0):
        # initial states of magnetization
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.m = numpy.array([self.x0, self.y0, self.z0])

        # hyper parameters
        self.damping_factor = 0.02
        self.gyo_ratio = 1.7e7  # unit Oe^-1 s^-1

        # declare variables
        self.external_field, self.anisotropy_field, self.demagnetization_field, self.dl_field = 0, 0, 0, 0
        self.effective_field = 0
        self.stt_amplitude = 0
        self.tao_1, self.tao_2, self.tao_3, self.tao_4 = None, None, None, None
        self.dc_frequency = 0

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
        self.effective_field = np.array([self.external_field+self.anisotropy_field*self.x0, 0,
                                         -self.demagnetization_field*self.z0])

        # STT term
        self.stt_amplitude = dc_amplitude + ac_amplitude*np.cos(time_ac*ac_frequency)

        # damping like field
        self.dl_field = - self.damping_factor / self.x0 * (
                self.external_field * self.x0 + self.anisotropy_field + self.demagnetization_field / 2)

        # the details of tao parameters from target paper
        self.tao_1 = -self.damping_factor * self.gyo_ratio * (
                self.x0 * self.external_field + self.anisotropy_field) - self.gyo_ratio * self.dl_field * self.x0

        self.tao_2 = -self.gyo_ratio * (
                self.external_field + self.anisotropy_field * self.x0 + self.demagnetization_field
                * self.x0) + self.gyo_ratio * self.damping_factor * self.dl_field

        self.tao_3 = self.gyo_ratio * (
                self.external_field + self.anisotropy_field*self.x0) - self.damping_factor*self.gyo_ratio*self.dl_field

        self.tao_4 = -self.damping_factor*self.gyo_ratio*(
            self.external_field*self.x0+self.anisotropy_field+self.demagnetization_field
        )-self.gyo_ratio*self.dl_field*self.x0

    def step_evolution(self):
        # time evolution
        self.m = numpy.array([self.x0, self.y0, self.z0])
        last_magnetization = self.m

        # K1 calculation Fourth Runge Kutta Method
        x_axis = np.array([1, 0, 0])

        delta_m = -self.gyo_ratio*np.cross(self.m, self.effective_field)-self.damping_factor*self.gyo_ratio*np.dot(
            self.m, self.effective_field)*self.m+self.damping_factor*self.gyo_ratio*self.effective_field+self\
            .stt_amplitude*self.damping_factor*self.gyo_ratio*(
                self.x0*np.cross(self.m, self.m)-np.cross(self.m, x_axis)+self.m*np.dot(self.m, x_axis)*self.m-x_axis)
        delta1_m_reduce = np.divide(delta_m, (1+pow(self.damping_factor, 2)))

        # K2
        self.m = last_magnetization + 1/2*delta1_m_reduce*t_step
        delta_m = -self.gyo_ratio * np.cross(self.m, self.effective_field) - self.damping_factor*self.gyo_ratio*np.dot(
            self.m, self.effective_field) * self.m + self.damping_factor * self.gyo_ratio * self.effective_field + \
            self.stt_amplitude*self.damping_factor*self.gyo_ratio * (
            self.x0 * np.cross(self.m, self.m) - np.cross(self.m,
                                                          x_axis) + self.m * np.dot(self.m, x_axis) * self.m - x_axis)
        delta2_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))

        # K3
        self.m = last_magnetization + 1 / 2 * delta2_m_reduce * t_step
        delta_m = -self.gyo_ratio * np.cross(self.m, self.effective_field)-self.damping_factor*self.gyo_ratio*np.dot(
            self.m, self.effective_field) * self.m + self.damping_factor * self.gyo_ratio * self.effective_field + \
            self.stt_amplitude * self.damping_factor*self.gyo_ratio * (
                          self.x0 * np.cross(self.m, self.m) - np.cross(self.m, x_axis) +
                          self.m * np.dot(self.m, x_axis) * self.m - x_axis)
        delta3_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))

        # K4
        self.m = last_magnetization + delta3_m_reduce * t_step
        delta_m = -self.gyo_ratio * np.cross(self.m, self.effective_field) - self.damping_factor*self.gyo_ratio*np.dot(
            self.m, self.effective_field) * self.m + self.damping_factor * self.gyo_ratio * self.effective_field + \
            self.stt_amplitude * self.damping_factor*self.gyo_ratio * (
                              self.x0 * np.cross(self.m, self.m) - np.cross(self.m, x_axis) +
                              self.m * np.dot(self.m, x_axis) * self.m - x_axis)
        delta4_m_reduce = np.divide(delta_m, (1 + pow(self.damping_factor, 2)))

        self.m = self.m + 1/6*t_step*(delta1_m_reduce+2*delta2_m_reduce+2*delta3_m_reduce+delta4_m_reduce)

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
        for i1 in range(int(time_consumed/time_step)):
            self.parameters_calculation(external_field, anisotropy_field, demagnetization_field, dc_amplitude,
                                        i1*time_step,
                                        ac_amplitude, ac_frequency)
            self.step_evolution()
            mx_list.append(self.x0)
            my_list.append(self.y0)
            mz_list.append(self.z0)
            t_list.append(i1*time_step)

        # find dc frequency
        extreme_points = []
        for i1 in range(len(mz_list)):
            if i1 != 0 and i1 != len(mz_list)-1:
                if mz_list[i1] < mz_list[i1-1] and mz_list[i1] < mz_list[i1+1]:
                    extreme_points.append(t_list[i1])
        self.dc_frequency = 1 / (extreme_points[int(len(extreme_points)/2)] - extreme_points[int(len(
            extreme_points)/2)-1])
        print('dc frequency: {}'.format(self.dc_frequency))

        return mx_list, my_list, mz_list, t_list


if __name__ == '__main__':
    # initial state
    a, b, c = 1, 1, 1
    t_step = 1e-9
    time_consume = 3e-7
    number_intervals = int(time_consume/t_step)
    extern_field = 200    # Unit: Oe
    ani_field = 0     # Unit: Oe
    dem_field = 8400    # Unit: Oe
    dc_current = 480
    ac_current = 50
    f_ac = 150e9

    mtj = Mtj(a, b, c)
    mx_list1, my_list1, mz_list1, t_list1 = mtj.frequency_dc(extern_field, ani_field, dem_field, dc_current,
                                                             ac_amplitude=ac_current, ac_frequency=f_ac,
                                                             time_consumed=time_consume, time_step=t_step)

    fourier_consequence = np.abs(np.fft.fft(mz_list1))
    pds = [i**2 for i in fourier_consequence]
    print(pds)

    plt.figure()
    plt.plot(mx_list1, c='red', label='mx')
    plt.legend()
    plt.ylabel('mx')
    plt.xlabel('Time:ns')

    plt.figure()
    plt.plot(my_list1, c='purple', label='my')
    plt.legend()
    plt.ylabel('my')
    plt.xlabel('Time:ns')

    plt.figure()
    plt.plot(mz_list1, c='orange', label='mz')
    plt.legend()
    plt.ylabel('mz')
    plt.xlabel('Time:ns')

    plt.figure('chaos')
    plt.scatter(my_list1, mz_list1, c='red')
    plt.ylabel('$m_Z$')
    plt.xlabel('$m_Y$')

    plt.figure('PDS')
    plt.plot(pds)

    plt.show()
