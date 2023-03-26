import numpy as np
import matplotlib.pyplot as plt
alpha = 3
betta = 0.002
delta = 0.0006
gamma = 0.5

def rk4(x_0, t_n, f, h):
    t = np.arange(0, t_n, h)
    n = len(t)
    res_x = np.zeros((n, 2))
    res_x[0] = x_0
    for i in range(n-1):
        k1 = h * f(res_x[i])
        k2 = h * f(res_x[i] + 0.5 * k1)
        k3 = h * f(res_x[i] + 0.5 * k2)
        k4 = h * f(res_x[i] + k3)
        res_x[i+1] = res_x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.
    return res_x


# Press the green button in the gutter to run the script.
def plot(arr, x0, y0):
    print(arr, x0,y0)
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    ax.plot(np.arange(0, 100, 0.1) ,arr[:,0],arr[:,1],  'blue') #,np.arange(0, 100, 0.1) arr[:,1] projection='2d'
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    ax.set_zlabel(r'$y$')
    plt.show()
    plt.savefig(f'base_{x0}_{y0}.png', dpi=300)
    plt.close()
def base():
    T = 1000
    def func(x):
        return np.array([alpha * x[0] - betta * x[0] * x[1], delta*x[0]*x[1] - gamma*x[1]])
    start_data = np.array([200*i for i in range(1, 11, 2)])
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    for x_0 in start_data:

        for y_0 in start_data:
            arr = rk4(np.array([x_0,y_0]), T, func, 0.1)

            ax.plot(arr[:, 0], arr[:, 1],
                    '-')  # ,np.arange(0, 100, 0.1) arr[:,1] projection='2d'
            ax.set_xlabel(r'$t$')
            ax.set_ylabel(r'$x$')

            #plot(rk4(np.array([1600,200]), 100, func, 0.1), 0, 0)
    ax.plot([0, 833.333], [0, 1500], 'ro')
    plt.savefig('base_fasa_dots.png', dpi=300)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    arr = rk4(np.array([400, 1600]), T, func, 0.01)

    ax.plot(np.arange(0, T, 0.01), arr[:, 1], 'blue', label = 'y(t)')
    ax.plot(np.arange(0, T, 0.01), arr[:, 0], 'red', label = 'x(t)' ) # ,np.arange(0, 100, 0.1) arr[:,1] projection='2d'
    ax.set_xlabel(r'$t$')
    plt.legend()
    plt.savefig('base_xy.png', dpi=300)
