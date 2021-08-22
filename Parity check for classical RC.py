import numpy as np
from matplotlib import pyplot as plt
from pyESN import ESN

# train_epoch >> 1 + input_size + reservoir_size
input_size = 1000
delay_time = 2
input_signal_raw = np.random.randint(0, 2, input_size).reshape(input_size, 1)
y_train_signal = input_signal_raw
if int(delay_time) != 0:
    for i2 in range(1, int(delay_time) + 1):
        temp_var1 = np.array(list(input_signal_raw)[-int(i2):] + list(input_signal_raw)[0:-int(i2)])
        y_train_signal = temp_var1 + y_train_signal
        y_train_signal[y_train_signal == 2] = 0

train_cutoff = int(np.ceil(0.9 * input_size))
print(train_cutoff)
y_train_signal = y_train_signal.reshape(-1, 1)
input_test, input_train = input_signal_raw[train_cutoff:], input_signal_raw[0:train_cutoff]
y_test, y_train = y_train_signal[train_cutoff:], y_train_signal[0:train_cutoff]

esn = ESN(n_inputs=1,
          n_outputs=1,
          n_reservoir=300,
          spectral_radius=0.3,
          density=0.5,
          noise=0.001,
          noise_ignore=True,
          input_shift=0,
          input_scaling=1,
          teacher_scaling=1,
          teacher_shift=0,
          teacher_forcing=False)

pre_train = esn.train_frequency(input_train, y_train)
pre_test = esn.test_parity(input_test)
print('error:{}'.format(np.var(pre_test-y_test)))

plt.figure('train')
x_sequence = np.linspace(0, len(y_train)-1, len(y_train))
plt.subplot(3, 1, 1)
plt.plot(x_sequence[0:100], input_train[0:100], label='input')
plt.ylabel('input')
plt.subplot(3, 1, 2)
plt.plot(x_sequence[0:100], y_train[0:100], label='target')
plt.ylabel('target')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(x_sequence[0:100], pre_train[0:100], label='model')
plt.ylabel('output')
plt.legend()

plt.figure('test')
plt.subplot(3, 1, 1)
plt.plot(x_sequence[0:100], input_test[0:100])
plt.ylabel('input')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(x_sequence[0:100], y_test[0:100], label='target', color='red')
plt.ylabel('target')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(x_sequence[0:100], pre_test[0:100], label='model', color='red')
plt.ylabel('model')
plt.legend()
plt.show()
