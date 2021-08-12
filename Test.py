import numpy as np
from matplotlib import pyplot as plt
from pyESN import ESN

# train_ctrl, train_output = frequency_control[:traintest_cutoff], frequency_output[:traintest_cutoff]
# test_ctrl, test_output = frequency_control[traintest_cutoff:], frequency_output[traintest_cutoff:]

input_size = 1000
delay_time = 1
input_signal_raw = np.random.randint(0, 2, input_size).reshape(1, input_size)
y_train_signal = input_signal_raw
if delay_time != 0:
    for i2 in range(1, delay_time + 1):
        temp_var1 = np.array(list(input_signal_raw)[-int(i2):] + list(input_signal_raw)[0:-i2])
        y_train_signal = temp_var1 + y_train_signal
        y_train_signal[y_train_signal == 2] = 0
# y_train_signal = y_train_signal.reshape(input_size, 1)
#
input_size = 100
input_signal_test = np.random.randint(0, 2, input_size)
y_test_signal = input_signal_test
if delay_time != 0:
    for i2 in range(1, delay_time + 1):
        temp_var1 = np.array(list(input_signal_test)[-int(i2):] + list(input_signal_test)[0:-i2])
        y_test_signal = temp_var1 + y_test_signal
        y_test_signal[y_test_signal == 2] = 0
# y_test_signal = y_test_signal.reshape(input_size, 1)

esn = ESN(n_inputs=1,
          n_outputs=1,
          n_reservoir=200,
          spectral_radius=0.25,
          density=0.5,
          noise=0.001,
          input_shift=[0],
          input_scaling=[0.01],
          teacher_scaling=1.12,
          teacher_shift=-0.7,
          out_activation=np.tanh,
          inverse_out_activation=np.arctanh)

pre_train = esn.train(input_signal_raw, y_train_signal)

print("test error:")
pre_test = esn.test(input_signal_raw)
print(np.sqrt(np.mean((pre_test - y_test_signal) ** 2)))

plt.figure()
# plt.plot(train_ctrl[window_tr, 1], label='control')
plt.plot(y_train_signal, label='target')
plt.plot(pre_train, label='model')
plt.legend(fontsize='x-small')
plt.title('training (excerpt)')
plt.ylim([-0.1, 1.1])

window_test = range(2000)
plt.figure(figsize=(10, 1.5))
# plt.plot(test_ctrl[window_test, 1], label='control')
plt.plot(y_train_signal, label='target')
plt.plot(pre_test, label='model')
plt.legend(fontsize='x-small')
plt.title('test (excerpt)')
plt.ylim([-0.1, 1.1])
plt.show()
