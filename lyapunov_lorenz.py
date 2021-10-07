import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from rich.progress import track

"""
this code is designed to calculate the lyapunov exponent function of Lorenz system.

We can package this function to judge edge of chaos in spintronic system.

The Lorenz System:

partial_x = sigma * (y - x)
partial_y = r*x - y - x*z
partial_z = x*y - batter*z

"""


def step_evolution(x0, y0, z0, r, t_step=0.01):
    # fourth Runge-Kutta
    sigma, beta = 10, 8/3
    h = t_step

    # k1
    k1_x = sigma*(y0-x0)
    k1_y = r*x0 - y0 - x0*z0
    k1_z = x0*y0 - beta*z0

    # k2
    k2_x = sigma*(y0-x0)
    k2_y = r*(x0 + h/2*k1_x) - (y0+h/2*k1_y) - (x0+h/2*k1_x)*(z0+h/2*k1_z)
    k2_z = (x0 + h/2*k1_x)*(y0+h/2*k1_y) - beta*(z0+h/2*k1_z)

    # k3
    k3_x = sigma * (y0 - x0)
    k3_y = r * (x0 + h / 2 * k2_x) - (y0 + h / 2 * k2_y) - (x0 + h / 2 * k2_x) * (z0 + h / 2 * k2_z)
    k3_z = (x0 + h / 2 * k2_x) * (y0 + h / 2 * k2_y) - beta * (z0 + h / 2 * k2_z)

    # k4
    k4_x = sigma * (y0 - x0)
    k4_y = r * (x0 + h * k3_x) - (y0 + h * k3_y) - (x0 + h * k3_x) * (z0 + h * k3_z)
    k4_z = (x0 + h * k3_x) * (y0 + h * k3_y) - beta * (z0 + h * k3_z)

    x_new = x0 + h/6*(k1_x + 2*k2_x + 2*k3_x + k4_x)
    y_new = y0 + h/6*(k1_y + 2*k2_y + 2*k3_y + k4_y)
    z_new = z0 + h/6*(k1_z + 2*k2_z + 2*k3_z + k4_z)
    return x_new, y_new, z_new


def time_evolution(x0=0.1, y0=0, z0=0, r=40, time_consume=0.1, t_step=0.0001):
    number_interval = int(time_consume/t_step)
    x_list, y_list, z_list = [], [], []
    x, y, z = x0, y0, z0
    for i in range(number_interval):
        x, y, z = step_evolution(x, y, z, r, t_step)
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return x_list, y_list, z_list


def lyapunov(r, x, y, z, perturbation):
    sigma, beta = 10, 8 / 3
    h1 = 0.002
    jacobian_matrix_raw = np.mat([[-sigma, sigma, 0], [r-z, -1, -x], [y, x, -beta]])

    # normalization
    perturbation = perturbation / np.linalg.norm(perturbation)

    # fourth-Runge-Kutta Method
    k1 = np.dot(jacobian_matrix_raw, perturbation)
    x_new_list, y_new_list, z_new_list = time_evolution(x, y, z, r, time_consume=h1/2)
    x_new, y_new, z_new = x_new_list[-1], y_new_list[-1], z_new_list[-1]
    jacobian_matrix = np.mat([[-sigma, sigma, 0], [r - z_new, -1, -x_new], [y_new, x_new, -beta]])
    perturbation_new = perturbation + h1/2*k1

    k2 = np.dot(jacobian_matrix, perturbation_new)
    x_new_list, y_new_list, z_new_list = time_evolution(x, y, z, r, time_consume=h1 / 2)
    x_new, y_new, z_new = x_new_list[-1], y_new_list[-1], z_new_list[-1]
    jacobian_matrix = np.mat([[-sigma, sigma, 0], [r - z_new, -1, -x_new], [y_new, x_new, -beta]])
    perturbation_new = perturbation + h1 / 2 * k2

    k3 = np.dot(jacobian_matrix, perturbation_new)
    x_new_list, y_new_list, z_new_list = time_evolution(x, y, z, r, time_consume=h1)
    x_new, y_new, z_new = x_new_list[-1], y_new_list[-1], z_new_list[-1]
    jacobian_matrix = np.mat([[-sigma, sigma, 0], [r - z_new, -1, -x_new], [y_new, x_new, -beta]])
    perturbation_new = perturbation + h1 * k3

    k4 = np.dot(jacobian_matrix, perturbation_new)
    perturbation_new = perturbation + h1/6*(k1+2*k2+2*k3+k4)

    delta_perturbation = np.dot(jacobian_matrix, perturbation_new).reshape(-1, 3)
    lyapunov_exponent = np.dot(delta_perturbation, perturbation_new) / pow(np.linalg.norm(perturbation_new), 2)

    return lyapunov_exponent, perturbation_new, x_new, y_new, z_new


def r_le_curve(r=28.0, x0=0.1, y0=0, z0=0, length_le=20):
    x, y, z = x0, y0, z0
    perturbation = np.random.rand(3, 1)

    last_error = np.inf
    while True:
        le_list_single = []
        for i in range(length_le):
            le1_single, perturbation, x, y, z = lyapunov(r, x, y, z, perturbation)
            le_list_single.append(le1_single[0, 0])
        le1 = sum(le_list_single)/length_le
        # print('############################')
        # print('standard deviation: {}'.format(np.std(le_list_single)))
        # print('lyapunov exponent: {}'.format(le1))
        # print('lyapunov_list :{}'.format(le_list_single))
        # print('#############################')
        if abs(np.std(le_list_single) - last_error) <= 0.01:
            break
        else:
            last_error = np.std(le_list_single)

    return le_list_single, x, y, z, perturbation, le1


if __name__ == '__main__':
    # verify the code of Lorenz Attractor behavior
    # mpl.rcParams["legend.fontsize"] = 10
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Lorenz Attractor')
    # xs, ys, zs = time_evolution(time_consume=100, r=28)
    # ax.plot3D(xs, ys, zs, label="Lorenz's strange attractor")
    # # plt.show()
    #
    # # bifurcation diagram
    # r_list = np.linspace(40, 100, 601)
    # r_maxes, z_maxes = [], []
    # r_min, z_min = [], []
    # for i1 in track(r_list):
    #     xs, ys, zs = time_evolution(time_consume=100, r=i1)
    #
    #     for i2 in range(1, len(zs)-1):
    #         if zs[i2] > zs[i2-1] and zs[i2] > zs[i2+1]:
    #             z_maxes.append(zs[i2])
    #             r_maxes.append(i1)
    #         elif zs[i2] < zs[i2-1] and zs[i2] < zs[i2+1]:
    #             z_min.append(zs[i2])
    #             r_min.append(i1)
    #     print(i1)
    #
    # plt.figure('bifurcation diagram')
    # plt.scatter(r_maxes, z_maxes, color="black", s=0.5, alpha=0.2)
    # plt.scatter(r_min, z_min, color="red", s=0.5, alpha=0.2)
    # # plt.show()

    # find lyapunov
    # _, _, _, _, _, le_1 = r_le_curve(r=94, length_le=1000)
    # print('le: {}'.format(le_1))
    r_list = np.linspace(96.1, 100, 40)
    # r_list = np.linspace(90, 100, 101)
    le_list_whole = []
    for i1 in r_list:
        _, _, _, _, _, le_1 = r_le_curve(r=i1, length_le=7000)
        le_list_whole.append(le_1)
        print('r={}, le={}'.format(i1, le_1))
        print('le_list_whole : {}'.format(le_list_whole))

    plt.figure()
    plt.plot(r_list, le_list_whole)
    plt.xlabel('r')
    plt.ylabel(r'lyapunov exponent')
    plt.show()
