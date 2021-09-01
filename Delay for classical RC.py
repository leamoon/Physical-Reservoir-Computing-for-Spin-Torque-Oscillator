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


def test_delay(input_size=1000, delay_time=0):
    """
    a function used to evaluate ability of classical echo state network (Linear ability)
    :param input_size: length of input signals
    :param delay_time: delay time
    :return: correlation between target and actual output
    """
    input_signal_raw = np.random.randint(0, 2, input_size)
    if delay_time != 0:
        y_train_signal = np.array(list(input_signal_raw)[-int(delay_time):] + list(input_signal_raw)[:-int(delay_time)])
    else:
        y_train_signal = input_signal_raw

    input_signal_raw = input_signal_raw.reshape(-1, 1)
    y_train_signal = y_train_signal.reshape(-1, 1)

    train_cutoff = int(np.ceil(0.9 * input_size))

    input_train, input_test = input_signal_raw[0:train_cutoff], input_signal_raw[-100:]
    target_train, target_test = y_train_signal[0:train_cutoff], y_train_signal[-100:]

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
    plt.savefig('./figures_classical/delay_delay={}'.format(delay_time))
    plt.close()

    # data processing
    pre_test, target_test = list(pre_test.T), list(target_test.T)
    while len(pre_test) == 1:
        pre_test, target_test = pre_test[0], target_test[0]
    # plt.show()
    cor_stm = pow(np.corrcoef(pre_test[int(delay_time):], target_test[int(delay_time):])[0, 1], 2)
    return cor_stm


if __name__ == '__main__':

    # calculate capacity
    cor_list = []
    for i in range(0, 21):
        cor = test_delay(input_size=1000, delay_time=i)
        cor_list.append(cor)
        print('-----------------------------------------------------')
        print('delay = {}'.format(i))
        print('-----------------------------------------------------')
    print(cor_list)
    print('capacity_STM:{}'.format(sum([i * i for i in cor_list])))

    plt.figure()
    delay_time_list = np.linspace(0, 20, 21)
    plt.plot(delay_time_list, cor_list)
    plt.scatter(delay_time_list, cor_list)
    plt.fill_between(delay_time_list, cor_list, alpha=0.5, color='red')
    plt.ylabel('quality')
    plt.xlabel('delay time')
    plt.text(1, 0.6, 'capacity={:.2f}'.format(sum([i * i for i in cor_list])), c='blue')
    plt.show()
