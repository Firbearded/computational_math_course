import random

import numpy as np
from matplotlib import pyplot as plt


def l_i(i, x, x_nodes):
    ans = 1
    for j in range(len(x_nodes)):
        if i != j:
            ans *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
    return ans


def L(x, x_nodes, y_nodes):
    ans = 0
    for i in range(len(y_nodes)):
        ans += y_nodes[i] * l_i(i, x, x_nodes)
    return ans


def func(x, a=[], b=[]):
    return 1 / (1 + 25 * x ** 2)


def plot_func(fig, ax, f, title, x_nodes, Xs, a=[], b=[]):
    y_nodes = np.array([f(x, a, b) for x in x_nodes])
    Ys_func = np.array([f(x, a, b) for x in Xs])
    Ys_L = np.array([L(x, x_nodes, y_nodes) for x in Xs])
    ax.plot(Xs, Ys_func, color='#aaa')
    ax.plot(Xs, Ys_L)
    ax.plot(x_nodes, y_nodes, 'ro', markersize=5)
    ax.legend(['f(x)', 'L(x)'])
    ax.set_title(title)
    ax.grid()


def easy_part():
    Ns = [5, 8, 11, 14, 17, 20, 23]
    for n in Ns:
        x_nodes_lin = np.linspace(-1, 1, n)
        Xs = np.linspace(-1, 1, 1000)

        fig1 = plt.figure(figsize=(6, 6))
        ax1 = plt.subplot()
        plot_func(fig1, ax1, func, "При равномерно-распределенных узлах:", x_nodes_lin, Xs)

        fig2 = plt.figure(figsize=(6, 6))
        x_nodes_cheb = np.array([(np.cos((2 * i - 1) / (2 * n) * np.pi)) for i in range(1, n + 1)])
        ax2 = plt.subplot()
        plot_func(fig2, ax2, func, "При чебышевских узлах:", x_nodes_cheb, Xs)
    plt.show()


def func_pade(x, a, b):
    up = 0
    down = 1
    for i in range(len(a)):
        up += a[i] * x ** i
    for k in range(1, len(b)):
        down += b[k] * x ** k
    return up / down


def hard_part():
    Xs = np.linspace(-1, 1, 500)
    f_num = 0
    while f_num < 6:
        f_num += 1
        n_f = random.randint(7, 15)
        m_f = random.randint(7, 15)
        a_f = []
        b_f = []
        for i in range(m_f + 1):
            a_f.append(random.random())
        for k in range(1, n_f + 1):
            b_f.append(random.random())
        not_smooth = False
        for x in Xs:
            down = 1
            for k in range(1, len(b_f)):
                down += b_f[k] * x ** k
            if down <= 0.0001:
                not_smooth = True
                continue
        if not_smooth:
            f_num -= 1
            continue

        Ns = []
        Rs1 = []
        Rs2 = []
        for n_nodes in range(1, 31):
            r1 = 0
            r2 = 0
            x_nodes_lin = np.linspace(-1, 1, n_nodes)
            y_nodes_lin = np.array([func_pade(x, a_f, b_f) for x in x_nodes_lin])
            x_nodes_cheb = np.array([(np.cos((2 * i - 1) / (2 * n_nodes) * np.pi)) for i in range(1, n_nodes + 1)])
            y_nodes_cheb = np.array([func_pade(x, a_f, b_f) for x in x_nodes_cheb])
            for x in Xs:
                r1 = max(r1, abs(func_pade(x, a_f, b_f) - L(x, x_nodes_lin, y_nodes_lin)))
                r2 = max(r2, abs(func_pade(x, a_f, b_f) - L(x, x_nodes_cheb, y_nodes_cheb)))
            # r = (max([abs(func_pade(x, a_f[f_num], b_f[f_num]) - L(x, x_nodes_lin, y_nodes_lin))]) for x in Xs)
            Rs1.append(r1)
            Rs2.append(r2)
            Ns.append(n_nodes)

        if f_num < 8:
            n_nodes = 10
            x_nodes_lin = np.linspace(-1, 1, n_nodes)
            x_nodes_cheb = np.array([(np.cos((2 * i - 1) / (2 * n_nodes) * np.pi)) for i in range(1, n_nodes + 1)])

            fig1 = plt.figure(figsize=(6, 6))
            ax1 = plt.subplot()
            plot_func(fig1, ax1, func_pade, "При равномерно-распределенных узлах:", x_nodes_lin, Xs, a_f, b_f)

            fig2 = plt.figure(figsize=(6, 6))
            ax2 = plt.subplot()
            plot_func(fig2, ax2, func_pade, "При чебышевских узлах:", x_nodes_cheb, Xs, a_f, b_f)

            fig3 = plt.figure(figsize=(6, 6))
            ax3 = plt.subplot()
            ax3.plot(Ns, Rs1)
            ax3.plot(Ns, Rs2)
            ax3.legend(['Расстояние при равномерно-распределенных узлах', 'Расстояние при чебышевских узлах'])
            ax3.set_yscale('log')
            ax3.grid()
    plt.show()


if __name__ == '__main__':
    hard_part()
