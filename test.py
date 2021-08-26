import numpy as np
import matplotlib.pyplot as plt

u0 = 12.56637e-7  # Vacuum permeability in H/m
h_bar = 1.054e-34  # Reduced Planck constant in Js
uB = 9.274e-24  # Bohr magneton in J/T
gamma = 2 * uB / (h_bar * 2 * np.pi)  # Gyromagnetic ratio in 1/Ts
alpha = 0.002
mx = -0.37
my = 0.0001
mz = 0.0001
Hx = 0.3  # applied filed along x axis
Hk = 0.3  # anisotropy field along x axis
Hd = 0.5  # demagnetization field along z axis
j = -1  # DC current density in MA/cm^2
time = 2e-6  # simulation time
t_step = 1e-11  # time step
step = time / t_step
n = int(step)
Matrix_mx = np.zeros(n)
Matrix_my = np.zeros(n)
Matrix_mz = np.zeros(n)
Matrix_t = np.zeros(n)
Matrix_mx[0] = mx
Matrix_my[0] = my
Matrix_mz[0] = mz
Matrix_t[0] = 0

# normalization
norm = np.sqrt(pow(my, 2)+pow(mx, 2)+pow(mz, 2))
mx = mx/norm
my = my/norm
mz = mz/norm


for i in range(1, n):
    t = i * t_step
    H_DL = 2000 * j  # field-like torque
    mx_1 = mx  # 0 = d_mx-gamma*Hd*my*mz-alpha*my*d_mz+alpha*mz*d_my-u0*gamma*H_DL*(my^2+mz^2);
    my_1 = my  # 0 = d_my+gamma*(Hx+Hk*mx+Hd*mx)*mz-alpha*mz*d_mx+alpha*mx*d_mz+u0*gamma*H_DL*mx*my;
    mz_1 = mz  # 0 = d_mz-gamma*(Hx+Hk*mx)*my-alpha*mx*d_my+alpha*my*d_mx+u0*gamma*H_DL*mx*mz;
    # k1
    A = gamma * Hd * my * mz
    B = u0 * gamma * H_DL * (my ** 2 + mz ** 2)
    C = -gamma * (Hx + Hk * mx + Hd * mx) * mz
    D = -u0 * gamma * H_DL * mx * my
    E = gamma * (Hx + Hk * mx) * my
    F = -u0 * gamma * H_DL * mx * mz
    a = 1 + (alpha ** 2) * (mz ** 2)
    b = alpha * my + (alpha ** 2) * mx * mz
    c = A + B - alpha * mz * (C + D)
    d = alpha * my - (alpha ** 2) * mx * mz
    e = 1 + (alpha ** 2) * (mx ** 2)
    f = E + F + alpha * mx * (C + D)
    d_mx1 = (b * f + c * e) / (a * e + b * d)
    d_mz1 = (a * f - c * d) / (a * e + b * d)
    d_my1 = C + D + alpha * mz * d_mx1 - alpha * mx * d_mz1
    mx = mx_1 + d_mx1 * t_step / 2
    my = my_1 + d_my1 * t_step / 2
    mz = mz_1 + d_mz1 * t_step / 2
    # k2
    A = gamma * Hd * my * mz
    B = u0 * gamma * H_DL * (my ** 2 + mz ** 2)
    C = -gamma * (Hx + Hk * mx + Hd * mx) * mz
    D = -u0 * gamma * H_DL * mx * my
    E = gamma * (Hx + Hk * mx) * my
    F = -u0 * gamma * H_DL * mx * mz
    a = 1 + (alpha ** 2) * (mz ** 2)
    b = alpha * my + (alpha ** 2) * mx * mz
    c = A + B - alpha * mz * (C + D)
    d = alpha * my - (alpha ** 2) * mx * mz
    e = 1 + (alpha ** 2) * (mx ** 2)
    f = E + F + alpha * mx * (C + D)
    d_mx2 = (b * f + c * e) / (a * e + b * d)
    d_mz2 = (a * f - c * d) / (a * e + b * d)
    d_my2 = C + D + alpha * mz * d_mx2 - alpha * mx * d_mz2
    mx = mx_1 + d_mx2 * t_step / 2
    my = my_1 + d_my2 * t_step / 2
    mz = mz_1 + d_mz2 * t_step / 2
    # k3
    A = gamma * Hd * my * mz
    B = u0 * gamma * H_DL * (my ** 2 + mz ** 2)
    C = -gamma * (Hx + Hk * mx + Hd * mx) * mz
    D = -u0 * gamma * H_DL * mx * my
    E = gamma * (Hx + Hk * mx) * my
    F = -u0 * gamma * H_DL * mx * mz
    a = 1 + (alpha ** 2) * (mz ** 2)
    b = alpha * my + (alpha ** 2) * mx * mz
    c = A + B - alpha * mz * (C + D)
    d = alpha * my - (alpha ** 2) * mx * mz
    e = 1 + (alpha ** 2) * (mx ** 2)
    f = E + F + alpha * mx * (C + D)
    d_mx3 = (b * f + c * e) / (a * e + b * d)
    d_mz3 = (a * f - c * d) / (a * e + b * d)
    d_my3 = C + D + alpha * mz * d_mx3 - alpha * mx * d_mz3
    mx = mx_1 + d_mx3 * t_step
    my = my_1 + d_my3 * t_step
    mz = mz_1 + d_mz3 * t_step
    # k4
    A = gamma * Hd * my * mz
    B = u0 * gamma * H_DL * (my ** 2 + mz ** 2)
    C = -gamma * (Hx + Hk * mx + Hd * mx) * mz
    D = -u0 * gamma * H_DL * mx * my
    E = gamma * (Hx + Hk * mx) * my
    F = -u0 * gamma * H_DL * mx * mz
    a = 1 + (alpha ** 2) * (mz ** 2)
    b = alpha * my + (alpha ** 2) * mx * mz
    c = A + B - alpha * mz * (C + D)
    d = alpha * my - (alpha ** 2) * mx * mz
    e = 1 + (alpha ** 2) * (mx ** 2)
    f = E + F + alpha * mx * (C + D)
    d_mx4 = (b * f + c * e) / (a * e + b * d)
    d_mz4 = (a * f - c * d) / (a * e + b * d)
    d_my4 = C + D + alpha * mz * d_mx4 - alpha * mx * d_mz4

    d_mx = d_mx1 + 2 * d_mx2 + 2 * d_mx3 + d_mx4
    d_my = d_my1 + 2 * d_my2 + 2 * d_my3 + d_my4
    d_mz = d_mz1 + 2 * d_mz2 + 2 * d_mz3 + d_mz4
    mx_1 = mx_1 + d_mx * t_step / 6
    my_1 = my_1 + d_my * t_step / 6
    mz_1 = mz_1 + d_mz * t_step / 6
    a = (mx_1 ** 2 + my_1 ** 2 + mz_1 ** 2) ** 0.5
    mx_1 = mx_1 / a
    my_1 = my_1 / a
    mz_1 = mz_1 / a

    mx = mx_1
    my = my_1
    mz = mz_1
    Matrix_mx[i] = mx
    Matrix_my[i] = my
    Matrix_mz[i] = mz
    Matrix_t[i] = i * t_step * 1e9

x = Matrix_mx
y = Matrix_t

plt.plot(y, x)
plt.xlabel('time(ns)')
plt.ylabel('mx')
axes = plt.gca()
axes.set_ylim([-1.5, 1.5])
plt.show()
