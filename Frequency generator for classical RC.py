import numpy as np
from matplotlib import pyplot as plt
from pyESN import ESN

rng = np.random.RandomState(42)


def frequency_generator(N, min_period, max_period, n_changepoints):
    """returns a random step function with N change points
       and a sine wave signal that changes its frequency at
       each such step, in the limits given by min_ and max_period."""
    # vector of random indices < N, padded with 0 and N at the ends:
    changepoints = np.insert(np.sort(rng.randint(0, N, n_changepoints)), [0, n_changepoints], [0, N])
    # list of interval boundaries between which the control sequence should be constant:
    const_intervals = list(zip(changepoints, np.roll(changepoints, -1)))[:-1]
    # populate a control sequence
    frequency_control = np.zeros((N, 1))
    for (t0, t1) in const_intervals:
        frequency_control[t0:t1] = rng.rand()
    periods = frequency_control * (max_period - min_period) + max_period
    # run time through a sine, while changing the period length
    frequency_output = np.zeros((N, 1))
    z = 0
    for i in range(N):
        z = z + 2 * np.pi / periods[i]
        frequency_output[i] = (np.sin(z) + 1) / 2
    return np.hstack([np.ones((N, 1)), 1 - frequency_control]), frequency_output


N = 30000  # signal length
min_period = 2
max_period = 10
n_changepoints = int(N / 200)
frequency_control, frequency_output = frequency_generator(N, min_period, max_period, n_changepoints)

traintest_cutoff = int(np.ceil(0.9 * N))
print(traintest_cutoff)

train_ctrl, train_output = frequency_control[:traintest_cutoff], frequency_output[:traintest_cutoff]
test_ctrl, test_output = frequency_control[traintest_cutoff:], frequency_output[traintest_cutoff:]
print(len(test_output))
print(train_ctrl)

esn = ESN(n_inputs=2,
          n_outputs=1,
          n_reservoir=300,
          spectral_radius=0.2,
          density=0.47,
          noise=0.0001,
          noise_ignore=True,
          input_shift=[0, 0],
          input_scaling=[-3, 3],
          teacher_scaling=0.2,
          teacher_shift=None,
          out_activation=np.tanh,
          inverse_out_activation=np.arctanh)

pred_train = esn.train_frequency(train_ctrl, train_output)

print("test error:")
pred_test = esn.test_frequency(test_ctrl)
print(np.sqrt(np.mean((pred_test - test_output) ** 2)))
plt.figure()
# plt.plot(train_ctrl[window_tr,1],label='control')
plt.plot(train_output, label='target')
plt.plot(pred_train, label='model')
plt.legend()
plt.title('training')
plt.ylim([-0.1, 1.1])

plt.figure()
# plt.plot(test_ctrl[window_test,1],label='control')
plt.plot(test_output, label='target')
plt.plot(pred_test, label='model')
plt.legend()
plt.title('test')
plt.ylim([-0.1, 1.1])
# plt.show()

plt.figure('frequency generator')
plt.subplot(3, 1, 1)
plt.plot(test_output[100:200], label='target', color='blue')
plt.title('Target')
# plt.ylabel('Target', rotation=0)

plt.subplot(3, 1, 2)
plt.plot(pred_test[100:200], label='model', color='orange')
plt.title('Output')
# plt.ylabel('Output', rotation=0)

plt.subplot(3, 1, 3)
plt.plot(test_output[100:200], label='target', color='blue')
plt.plot(pred_test[100:200], label='model', color='orange')
plt.title('Mix')
# plt.ylabel('Mix', rotation=0)
plt.legend()
plt.show()
