import numpy as np
from matplotlib import pyplot as plt
from pyESN import ESN

esn = ESN(n_inputs=1,
          n_outputs=1,
          n_reservoir=400,
          spectral_radius=0.3,
          density=0.5,
          noise=0.0001,
          noise_ignore=False,
          input_shift=None,
          input_scaling=None,
          teacher_scaling=None,
          teacher_shift=None,
          teacher_forcing=False)

# train_epoch >> 1 + input_size + reservoir_size
input_size = 10000
delay_time = 2
index_slide = input_size/4*3
input_signal_raw = np.random.randint(0, 2, input_size)
if delay_time != 0:
    y_train_signal = np.array(list(input_signal_raw)[-int(delay_time):] + list(input_signal_raw)[:-int(delay_time)])
else:
    y_train_signal = input_signal_raw

input_signal_raw = input_signal_raw.reshape(-1, 1)
y_train_signal = y_train_signal.reshape(-1, 1)

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
plt.plot(x_sequence[0:100], input_train[0:100], label='input')
plt.ylabel('input')
plt.subplot(3, 1, 2)
plt.plot(x_sequence[0:100], target_train[0:100], label='target')
plt.ylabel('target')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(x_sequence[0:100], pre_train[0:100], label='model')
plt.ylabel('output')
plt.legend()

plt.figure('test')
plt.subplot(3, 1, 1)
plt.plot(input_test[0:100])
plt.subplot(3, 1, 2)
plt.plot(target_test[0:100], label='target')
plt.subplot(3, 1, 3)
plt.plot(pre_test[0:100], label='model')
plt.show()
