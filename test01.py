from nolitsa import data, lyapunov
import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
x0 = [0.62225717, -0.08232857, 30.60845379]
x = data.lorenz(length=4000, sample=dt, x0=x0,
                sigma=16.0, beta=4.0, rho=45.92)[1]
plt.plot(range(len(x)), x)

# Choose appropriate Theiler window.
meanperiod = 30
maxt = 250
d = lyapunov.mle(x, maxt=maxt, window=meanperiod)
t = np.arange(maxt) * dt

plt.figure()
plt.title('Maximum Lyapunov exponent for the Lorenz system')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Average divergence $\langle d_i(t) \rangle$')
plt.plot(t, d, label='divergence')
plt.legend()
plt.show()
