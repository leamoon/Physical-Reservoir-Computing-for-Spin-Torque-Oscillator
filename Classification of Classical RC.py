import numpy as np
from matplotlib import pyplot as plt
from pyESN import ESN

esn = ESN(n_inputs=1,
          n_outputs=1,
          n_reservoir=50,
          spectral_radius=0.3,
          density=0.5,
          noise=0.001,
          noise_ignore=True,
          input_shift=None,
          input_scaling=None,
          teacher_scaling=None,
          teacher_shift=None,
          teacher_forcing=False)


def waveform_generator(current_limited, number_wave):
    """
    a function used to create random wave(sine or square)
    :param current_limited: the range of voltages
    :param number_wave: number of waves
    :return: wave_points
    """
    # 8 points to express waveform
    random_pulse = np.random.randint(0, 4, int(number_wave))
    print('random:{}'.format(random_pulse))
    wave_points = []
    for i in random_pulse:
        if i == 0:
            # sine function
            sine_points = [0, current_limited / 2, current_limited, current_limited / 2, 0,
                           -current_limited / 2, -current_limited, -current_limited / 2]
            wave_points = wave_points + sine_points
        elif i == 1:
            # square function
            square_points = [current_limited] * 4 + [-current_limited] * 4
            wave_points = wave_points + square_points
        elif i == 2:
            # cos function
            cos_points = [0, -current_limited / 2, -current_limited, -current_limited / 2, 0,
                          current_limited / 2, current_limited, current_limited / 2]
            wave_points = wave_points + cos_points
        elif i == 3:
            # triangle function
            tri_points = [0, current_limited / 3, 2*current_limited/3, current_limited, current_limited,
                          current_limited, 2*current_limited/3, current_limited / 3]
            wave_points = wave_points + tri_points

    print('wave:{}'.format(wave_points))

    # train_signal = []
    # for ac_value in random_pulse:
    #     temp = [ac_value] * 8
    #     train_signal = train_signal + temp
    # print('train_signal:{}'.format(train_signal))
    return wave_points, list(random_pulse)


def test_delay(input_size=300, delay_time=0):
    """
    a function used to evaluate ability of classical echo state network (Linear ability)
    :param input_size: length of input signals
    :param delay_time: delay time
    :return: correlation between target and actual output
    """
    input_signal_raw, y_train_signal = waveform_generator(current_limited=1, number_wave=input_size)

    input_signal_raw = np.array(input_signal_raw).reshape(-1, 1)
    y_train_signal = np.array(y_train_signal).reshape(-1, 1)

    train_cutoff = int(np.ceil(0.9 * input_size))

    input_train, input_test = input_signal_raw[0:train_cutoff], input_signal_raw[train_cutoff:]
    target_train, target_test = y_train_signal[0:train_cutoff], y_train_signal[train_cutoff:]

    # train
    pre_train = esn.train_parity(input_train, target_train)
    pre_test = esn.test_frequency(input_test)
    print('test_error:{}'.format(np.var(target_test-pre_test)))

    plt.figure('train')
    x_sequence = np.linspace(0, len(target_train)-1, len(target_train))
    plt.subplot(3, 1, 1)
    plt.plot(x_sequence[0:100], input_train[0:100], label='input', color='red')
    plt.ylabel('input')
    plt.subplot(3, 1, 2)
    plt.plot(x_sequence[0:100], target_train[0:100], label='target')
    plt.ylabel('target')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(x_sequence[0:100], pre_train[0:100], label='model')
    plt.ylabel('output')
    plt.legend()
    plt.close()

    plt.figure('test')
    plt.subplot(3, 1, 1)
    plt.plot(input_test[0:100])
    plt.ylabel('input_signals')
    plt.subplot(3, 1, 2)
    plt.plot(target_test[0:100], label='target', color='red')
    plt.ylabel('target')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(pre_test[0:100], label='model', color='red')
    plt.ylabel('model')
    plt.legend()
    plt.show()

    return 0


if __name__ == '__main__':

    # calculate capacity
    # cor_list = []
    # for ac_value in range(0, 21):
    #     cor = test_delay(input_size=1000, delay_time=ac_value)
    #     cor_list.append(cor)
    #     print('-----------------------------------------------------')
    #     print('delay = {}'.format(ac_value))
    #     print('-----------------------------------------------------')
    # print(cor_list)
    # print('capacity_STM:{}'.format(sum([ac_value * ac_value for ac_value in cor_list])))
    #
    # plt.figure()
    # delay_time_list = np.linspace(0, 20, 21)
    # plt.plot(delay_time_list, cor_list)
    # plt.scatter(delay_time_list, cor_list)
    # plt.fill_between(delay_time_list, cor_list, alpha=0.5, color='red')
    # plt.ylabel('quality')
    # plt.xlabel('delay time')
    # plt.text(1, 0.6, 'capacity={:.2f}'.format(sum([ac_value * ac_value for ac_value in cor_list])), c='blue')
    # plt.show()
    test_delay()
