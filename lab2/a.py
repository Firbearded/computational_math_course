import numpy as np
import matplotlib.pyplot as plt


def g1(x):
    return x*np.exp(x)


def g2(x):
    return x*x*np.sin(3*x)


def diff2(x_0, h, f):
    return (f(x_0+h)-f(x_0-h))/(2*h)


def diff4(x_0, h, f):
    return (f(x_0-2*h)-8*f(x_0-h)+8*f(x_0+h)-f(x_0-2*h))/(12*h)


def err_dif2(x_0, h):
    g1_d_real = np.exp(x_0) + x_0 * np.exp(x_0)
    g1_d = diff2(x_0, h, g1)
    return abs(g1_d - g1_d_real)


def err_dif4(x_0, h):
    g1_d_real = np.exp(x_0) + x_0 * np.exp(x_0)
    g1_d = diff4(x_0, h, g1)
    return abs(g1_d - g1_d_real)


def base_diff():
    x_0 = 2
    h = np.logspace(-16, 0, 100)
    #график доделать по правилам
    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    ax.grid()
    ax.set_xlabel('$h$')
    ax.set_ylabel('$\delta(g_1\'(x_0))$')
    h_for_scaling = np.logspace(-5, 0, 100)
    ax.loglog(h, [err_dif(x_0, el) for el in h ], 'ro', label = '$\delta(g_1\'(x_0))$')
    ax.loglog(h_for_scaling, h_for_scaling ** 2, '--',label = '$O(h^2)$' )
    ax.legend()
    plt.show()


def composite_simpson(a, b, n, f):
    x = np.linspace(a, b, n)
    h = (b-a)/(n-1)
    return h/3 *(f(x[0]) + 2*np.sum(f(x[2:-1:2])) + 4*np.sum(f(x[1::2])) + f(x[-1]))


def err_int(a, b, n):
    g2_int_real = (-b ** 2 * np.cos(3 * b) + a ** 2 * np.cos(3 * a)) / 3 + (2 * b * np.sin(3 * b) - 2 * a * np.sin(3 * a)) / 9 + (2 * np.cos(3 * b) - 2 * np.cos(3 * a)) / 27
    g2_int = composite_simpson(a, b, n, g2)
    return np.abs(g2_int_real - g2_int)


def base_int():
    a = 0
    b = np.pi
    n1 = 3
    n2 = 9999
    h1 = np.log10((b - a) / (n1 - 1))
    h2 = np.log10((b - a) / (n2 - 1))
    h = np.logspace(h2, h1, 100)
    n = np.array((b - a) / h + 1, dtype=int)
    print(n)
    for i in range(len(n)):
        if n[i] % 2 == 0:
            n[i] += 1
    h = np.array((b - a) / (n - 1))
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.grid()
    ax.set_xlabel('$h$')
    ax.set_ylabel('$Δ$')
    ax.loglog(h, [err_int(a, b, el) for el in n], 'ro', markersize=1, label = 'Абсолютная погрешность численного интегрирования')
    h_for_scaling = np.logspace(h1, h2/2, 100)
    ax.loglog(h_for_scaling, 0.2 * h_for_scaling ** 4, label = '$O(h^4)$')
    ax.legend(['Абсолютная погрешность численного интегрирования', 'O(h^4)'])
    ax.legend()
    plt.show()




if (__name__ == '__main__'):
    #base_diff()
    base_int()

