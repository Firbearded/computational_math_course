import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return np.array([3 * x[0] - 0.002 * x[0] * x[1], 0.0006 * x[0] * x[1] - 0.5 * x[1]])


def r_k4(x_0, t_n, f, h):
    t = np.array([i * h for i in range(0, int(t_n // h) + 2)])
    w = [x_0]
    for i in range(len(t) - 1):
        k1 = h * f(w[i])
        k2 = h * f(w[i] + 1 / 2 * k1)
        k3 = h * f(w[i] + 1 / 2 * k2)
        k4 = h * f(w[i] + k3)
        w.append(w[i] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return t, w


def lu(A):
    n = len(A)
    C = np.array(A.copy())
    for i in range(n):
        for j in range(i + 1, n):
            C[j][i] /= C[i][i]
            for k in range(i + 1, n):
                C[j][k] -= C[j][i] * C[i][k]
    U = np.triu(C)
    L = np.tril(C, -1)
    return L, U


def lu_perm(A: list, permute: bool):
    n = len(A)
    C = np.array(A.copy())
    P = np.array([np.array([0**(abs(i-j)) for j in range(n)]) for i in range(n)], dtype=np.float64)
    for i in range(n):
        max_abs = 0
        max_row = -1
        for j in range(i, n):
            if(abs(C[j][i]) > max_abs):
                max_abs = abs(C[j][i])
                max_row = j
        if(max_abs!=0):
            if(permute):
                P[[max_row, i]] = P[[i, max_row]]
                C[[max_row, i]] = C[[i, max_row]]
            for j in range(i+1, n):
               C[j][i] /= C[i][i]
               for k in range(i+1, n):
                   C[j][k] -= C[j][i] * C[i][k]
    U = np.triu(C)
    L = np.tril(C, -1)
    return P, L, U



def solve(L, U, P, b):
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    b = np.matmul(P, b)
    for i in range(n):
        y[i] = b[i] - np.sum([L[i][k] * y[k] for k in range(i)])
    for i in range(n):
        x[n - i - 1] = (y[n - i - 1] - np.sum([U[n - i - 1][n - k - 1] * x[n - k - 1] for k in range(i)])) / \
                       U[n - i - 1][n - i - 1]
    return x


def J(x):
    return np.array([
        np.array([3 - 0.002 * x[1], -0.002 * x[0]]),
        np.array([0.0006 * x[1], 0.0006 * x[0] - 0.5])
    ])


def newton(x_0, f, J):
    count = 1
    J_ = J(x_0)
    P, L, U = lu_perm(J_, 1)
    b = np.array([[f(x_0)[0]], [f(x_0)[1]]], dtype=np.float64)
    y = solve(L, U, P, b)
    lamb = []

    x_1 = x_0 - y
    while abs(np.linalg.norm(x_0, ord=np.inf) - np.linalg.norm(x_1, ord=np.inf)) > 0.00000001:
        x_0 = x_1
        J_ = J(x_0)
        P, L, U = lu_perm(J_, 1)
        y = solve(L, U, P, f(x_0))
        x_1 = x_0 - y
        lamb.append(
            np.abs(np.linalg.norm(x_1, ord=np.inf) - np.linalg.norm(np.array([833.333, 1500]), ord=np.inf)) / np.abs(
                np.linalg.norm(x_0, ord=np.inf) - np.abs(np.linalg.norm(np.array([833.333, 1500]), ord=np.inf))) ** 2)
        count += 1

    return x_1, count, lamb


def get_t(f, x_0, z):
    def g(f, x_0):
        return np.dot(f(x_0).transpose(), f(x_0))
    def h(t, x_0, z, f):
        return g(f, x_0 - t * z / np.linalg.norm(z, ord=2))
    t1 = 0
    t3 = 1
    h1 = h(t1, x_0, z, f)
    h3 = h(t3, x_0, z, f)
    flag = 0
    if abs(h(t3 / 4, x_0, z, f) - h1) <= abs(h(t3 * 4, x_0, z, f) - h1):
        flag = 1
    while h3 > h1:
        if flag:
            t3 /= 4
        else:
            t3 *= 4
        h3 = h(t3, x_0, z, f)
    t2 = t3 / 2.
    h2 = h(t2, x_0, z, f)
    a = h1 / (t1 - t2) / (t1 - t3)
    b = h2 / (t2 - t1) / (t2 - t3)
    c = h3 / (t3 - t1) / (t3 - t2)
    t = (a * (t2 + t3) + b * (t1 + t3) + c * (t1 + t2)) / 2 / (a + b + c)

    return t


def gradient_descent(x_0, f, J):

    count = 0
    x_1 = x_0
    fl = 1
    lamb = []
    while abs(np.linalg.norm(x_0, ord=np.inf) - np.linalg.norm(x_1, ord=np.inf)) > 0.00000001 or fl:
        fl = 0

        x_0 = x_1
        J_ = J(x_0)
        z = np.dot(J_.transpose(), f(x_0))

        t = get_t(f, x_0, z)

        x_1 = x_0 - t * z / np.linalg.norm(z, ord=2)
        lamb.append(
            np.abs(np.linalg.norm(x_1, ord=np.inf) - np.linalg.norm(np.array([833.333, 1500]), ord=np.inf)) / np.abs(
                np.linalg.norm(x_0, ord=np.inf) - np.abs(np.linalg.norm(np.array([833.333, 1500]), ord=np.inf))))
        count += 1
    return x_1, count, lamb



def main():
    h = 0.2
    t_n = 18
    supr_norm1 = []
    supr_norm2 = []
    sum1 = 0
    sum2 = 0
    sq_otk1 = 0
    sq_otk2 = 0
    arr_c1 = []
    arr_c2 = []
    Xx = []
    Yy = []
    for i in range(0, 201):
        supr_norm1.append([])
        supr_norm2.append([])
        Yy.append(i)
        Xx.append(i)
    ii = 0
    for i in range(0, 201):
        for j in range(0, 201):
            x1, c1, lamb1 = newton(np.array([15 * i, 15 * j]), f, J)
            arr_c1.append(c1)
            sum1 += c1
            x2, c2, lamb2 = gradient_descent(np.array([15 * i, 15 * j]), f, J)
            sum2 += c2
            arr_c2.append(c2)
            supr_norm1[ii].append(np.linalg.norm(x1, ord=np.inf))
            supr_norm2[ii].append(np.linalg.norm(x2, ord=np.inf))
            # print(x1, '=', x2)
            # print(c1, ', ', c2)
        ii += 1
    m1 = sum1 / 201 / 201
    m2 = sum2 / 201 / 201
    for c in arr_c1:
        sq_otk1 += (c - m1) ** 2 / (201 * 201 - 1)
    for c in arr_c2:
        sq_otk2 += (c - m2) ** 2 / (201 * 201 - 1)
    sq_otk1 = sq_otk1 ** (1 / 2)
    sq_otk2 = sq_otk2 ** (1 / 2)
    print("Мат ожидание количества итераций метода Ньютона:", m1)
    print("Средне квадаратичное отклонение количества итераций метода Ньютона:", sq_otk1)
    print("Мат ожидание количества итераций метода градиентного спуска:", m2)
    print("Средне квадаратичное отклонение количества итераций метода градиентного спуска:", sq_otk2)
    
    figp1 = plt.figure()
    axp1 = plt.subplot()
    axp1.set_xlabel('$x_i^{(0)}$')
    axp1.set_ylabel('$y_i^{(0)}$')
    axp1.contourf(Xx, Yy, supr_norm1)

    figp2 = plt.figure()
    axp2 = plt.subplot()
    axp2.set_xlabel('$x_i^{(0)}$')
    axp2.set_ylabel('$y_i^{(0)}$')
    axp2.contourf(Xx, Yy, supr_norm2)
    plt.show()

    """c, x, k = gradient_descent(np.array([2000, 300]), f, J)
    x_data_prow = np.linspace(1e-4, 1e2, 30)
    y_data_prow = x_data_prow
    x_data = np.logspace(-4, 2, 10)
    y_data = k * x_data
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data, 'bo', label=r'$y_{grad}=\lambda x$')
    ax.plot(x_data_prow, y_data_prow, 'red', label=r'$O(x)$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    c, x, k = newton(np.array([500, 1400]), f, J)
    x_data_prow = np.linspace(1e-4, 1e2, 30)
    y_data_prow = x_data_prow ** 2
    x_data = np.logspace(-4, 2, 30)
    y_data = k * x_data ** 2
    plt.savefig('gradient_loglog.png', dpi=300)
    plt.legend()
    fig, ax = plt.subplots()
    ax.plot(x_data_prow, y_data_prow, 'blue', label=r'$O(x^2)$')
    ax.plot(x_data, y_data, 'ro', label=r'$y_{newton}=\lambda x$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend()
    plt.show()"""

    fig = plt.figure()
    ax = plt.subplot()
    flag = 1
    for i in range(1, 11):
        for j in range(1, 11):
            t, w = r_k4(np.array([200 * i, 200 * j]), t_n, f, h)
            x = []
            y = []
            for w_i in w:
                x.append(w_i[0])
                y.append(w_i[1])
            if flag:
                fig1 = plt.figure()
                flag = 0
                ax1 = plt.subplot()
                ax1.plot(t, x, label = '$x(t)$')
                ax1.plot(t, y, label = '$y(t)$')
                ax1.set_xlabel('$t$')
                ax1.legend()
                #ax1.plot([0, t_n], [2500 / 3, 2500 / 3])  # происходит точка перегиба или как там оно называется
                #ax1.plot([0, t_n], [1500, 1500])  # значение 2 производной меняет знак (но почему второй, если мы первую к 0 прировняли??)
            ax.plot(x, y)
    ax.plot([0, 2500 / 3], [0, 1500], 'ro', markersize=5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.show()

if __name__ == '__main__':
    main()
