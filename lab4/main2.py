import matplotlib.pyplot as plt
import numpy as np
import os
from base_file import *
import time
from multiprocessing import Pool

#b = np.eye(2)
def f(x):
    return np.array([alpha * x[0] - betta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]])


# f = lambda x:
def h(t, x_k, z_k):
    return f(x_k - t * z_k / (np.linalg.norm(z_k, ord=2))).dot(f(x_k - t * z_k / (np.linalg.norm(z_k, ord=2))))
def lu(A: list, permute: bool):
    n = len(A)
    p = []
    C = np.array(A.copy())
    P = np.array([np.array([0.**(abs(i-j)) for j in range(n)]) for i in range(n)])
    for i in range(n):
        pivotValue = 0
        pivot = -1
        for row in range(i, n):
            if( abs(C[ row ][ i ]) > pivotValue ):
                pivotValue = abs(C[ row ][ i ])
                pivot = row
        if( pivotValue != 0 ):
            if(permute):
                P[[pivot, i]] = P[[i, pivot]]
                C[[pivot, i]] = C[[i, pivot]]
            for j in range(i+1, n):
               C[ j ][ i ] /= C[ i ][ i ]
               for k in range(i+1, n):
                   C[ j ][ k ] -= C[ j ][ i ] * C[ i ][ k ]
    L = np.array([np.array([0. for j in range(n)]) for i in range(n)])
    U = np.array([np.array([0. for j in range(n)]) for i in range(n)])
    for i in range(n):
        for j in range(n):
            if(i>j):
                L[i][j] = C[i][j]
            if(i==j):
                L[i][j] = 1
                U[i][j] = C[i][j]
            if(i<j):
                U[i][j] = C[i][j]
    return L, U, P

def solve(L, U,P, b):
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    b = np.matmul(P, b)
    for i in range(n):
        y[i] = b[i] - sum([L[i][k] * y[k] for k in range(i)])
    for i in range(n):
        x[n - i - 1] = (y[n - i - 1] - sum([U[n - i - 1][n - k - 1] * x[n - k - 1] for k in range(i)])) / \
                       U[n - i - 1][n - i - 1]
    return x

def newtonV2(x_0, f, J):
    E = 1e-8
    x = []
    x.append(x_0)
    L, U, P = lu(J(x_0), True)
    Y = solve(L,U,P,f(x_0))
    x_next = x_0 - Y
    x.append(x_next)
    i = 1
    k = []
    k_x_k = np.array([0, 0])
    k_x_1 = np.array([0, 0])
    while(abs(np.linalg.norm(x[i]-x[i-1], ord=np.inf)) > E):
        L, U, P = lu(J(x[i]), True)
        Y = solve(L, U, P, f(x[i]))

        k_x_1 = x[i-1]
        k_x_k = x[i]
        k.append(
            np.abs(np.linalg.norm(k_x_k, ord=np.inf) - np.linalg.norm(np.array([833.333, 1500]), ord=np.inf)) / np.abs(
                np.linalg.norm(k_x_1, ord=np.inf) - np.abs(np.linalg.norm(np.array([833.333, 1500]), ord=np.inf)))**2)
        x_next = x[i] - Y
        x.append(x_next)
        i += 1
    #k = np.abs(np.linalg.norm(k_x_k, ord=np.inf) - np.linalg.norm(x[-1], ord=np.inf)) / (np.abs(
    #    np.linalg.norm(k_x_1, ord=np.inf) - np.abs(np.linalg.norm(x[-1], ord=np.inf))))**2
    return len(x)-1, x[-1], np.array(k)

def tSearch(x_k, z_k):

    #h = lambda t:
    t = np.linspace(-8, 2, num=20)
    h_k = np.array([h(10**i,x_k, z_k) for i in t])
    t_res = 10**t[np.argmin(h_k)]
    #print(t_res)
    return t_res


def resSearch(t_k, x_k, z_k):

    x_k = x_k - t_k * z_k / (np.linalg.norm(z_k, ord=2))
    return x_k

def gradient_descent_M(x_0, f, J):
    eps = 1E-8
    x_1 = x_0
    J_t = np.transpose(J(x_1))
    z_k = J_t.dot(f(x_1))
    t = tSearch(x_1, z_k)
    x_k = resSearch(t, x_1, z_k)
    count = 1
    k = []
    k_x_k = np.array([0,0])
    k_x_1 = np.array([0,0])
    while np.linalg.norm(x_k-x_1, ord=np.inf)> eps:
        count += 1

        k_x_1 = x_1
        k_x_k = x_k
        k.append(
            np.abs(np.linalg.norm(k_x_k, ord=np.inf) - np.linalg.norm(np.array([833.333, 1500]), ord=np.inf)) / np.abs(
                np.linalg.norm(k_x_1, ord=np.inf) - np.abs(np.linalg.norm(np.array([833.333, 1500]), ord=np.inf))))

        x_1 = x_k
        J_t = np.transpose(J(x_1))
        z_k = J_t.dot(f(x_1))
        t_res = tSearch(x_1, z_k)
        x_k = resSearch(t_res, x_1, z_k)
    #k.append(np.abs(np.linalg.norm(k_x_k, ord=np.inf)-np.linalg.norm(np.array([833.333, 1500]), ord=np.inf))/np.abs(np.linalg.norm(k_x_1, ord=np.inf)-np.abs(np.linalg.norm(np.array([833.333, 1500]), ord=np.inf))))
    return count, x_k, np.array(k)


def trans(J):
    n = len(J)
    res = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            res[j][i] = J[i][j]
    return res


def adv_5f():
    def J(x):
        res = np.array([[0.0,0.0],[0.0,0.0]])

        res[0][0] = alpha - betta*x[1]
        res[0][1] = -betta*x[0]
        res[1][0] = delta*x[1]
        res[1][1] = delta*x[0] - gamma
        return res
    c, x, k = gradient_descent_M(np.array([1800, 400]), f, J)
    x_data_prow = np.linspace(1e-5, 1e2, 30)
    y_data_prow = x_data_prow
    x_data = np.logspace(-5, 2, len(k))
    y_data = k*x_data
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data, 'bo', label = r'$y_{grad}=\lambda x$')
    ax.plot(x_data_prow, y_data_prow, 'red', label = r'$O(x)$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    c, x, k = newtonV2(np.array([1800, 400]), f, J)

    x_data_prow = np.linspace(1e-5, 1e2, 30)
    y_data_prow = x_data_prow**2
    x_data = np.logspace(-5, 2, len(k))
    y_data = k*x_data ** 2
    plt.legend()
    plt.grid()
    plt.savefig('gradient_loglog.png', dpi=300)

    fig, ax = plt.subplots()
    ax.plot(x_data_prow, y_data_prow, 'blue', label = r'$O(x^2)$')
    ax.plot(x_data, y_data, 'ro',label = r'$y_{newton}=\lambda x$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend()
    plt.grid()
    plt.savefig('newton_loglog.png', dpi=300)
    plt.show()


def advanced():
    p= Pool()
    def J(x):
        res = np.array([[0.0,0.0],[0.0,0.0]])

        res[0][0] = alpha - betta*x[1]
        res[0][1] = -betta*x[0]
        res[1][0] = delta*x[1]
        res[1][1] = delta*x[0] - gamma
        return res
    def f(x):
        return np.array([alpha * x[0] - betta * x[0] * x[1], delta*x[0]*x[1] - gamma*x[1]])
    start = time.time()
    start_data = np.array([15 * i for i in range(0, 201)])
    count = 0
    n = len(start_data)
    #print(newtonV2(np.array([200, 200]), f, J))
    res = np.zeros((n, n))
    res2 = np.zeros((n, n))
    fig, ax = plt.subplots()
    sum_grad = []
    sum_newton = []
    in_range = []
    for x_0 in range(n):
        for y_0 in range(n):
            count += 1
            os.system('CLS')
            ne_start = time.time()
            print(f"Осталось: {100 - count/(n**2)*100 :.{1}f}%, прошло {(ne_start - start)/60:.{3}f} минут, ")
            #print(count, f')x = {start_data[x_0]}y ={start_data[y_0]}---------')

            c, x, k = newtonV2(np.array([start_data[x_0], start_data[y_0]]), f, J)
            sum_newton.append(c)
            res2[x_0][y_0] = np.linalg.norm(x, ord=np.inf)
            #print('newton', c, np.linalg.norm(x, ord=np.inf)) #
            c, x, k = gradient_descent_M(np.array([start_data[x_0], start_data[y_0]]), f, J)
            sum_grad.append(c)
            res[x_0][y_0] = np.linalg.norm(x, ord=np.inf)

            #print('grad', c, np.linalg.norm(x, ord=np.inf))
    M_newton = sum(sum_newton)/count
    M_grad = sum(sum_grad)/count
    print("Мат. ожидание для ньютона:", M_newton, "Мат. ожидание для градиента:", M_grad)
    Sr_newton = 0
    Sr_grad = 0
    for i in range(len(sum_newton)):
        Sr_grad += (sum_grad[i] - M_grad)**2/(len(sum_newton)*(len(sum_newton)-1))
        Sr_newton += (sum_newton[i] - M_newton) ** 2 / (len(sum_newton) * (len(sum_newton) - 1))
    print("Среднеквадратичное отклонение для ньютона:",Sr_newton**0.5, "Среднеквадратичное отклонение для градиента:", Sr_grad**0.5)
    x_range = np.linspace(0, 3000, n)
    y_range = np.linspace(0, 3000, n)
    cs = ax.contourf(x_range, y_range,res, cmap ="autumn")
    cbar = plt.colorbar(cs)
    #ax.clabel(cs)
    plt.savefig('grad_res_modifi_1.png', dpi=300)
    # print('res')
    # for i in res:
    #     for j in i:
    #         print(j, end=' ')
    #     print('\n')
    fig, ax = plt.subplots()
    cs  = ax.contourf(x_range, y_range, res2, cmap ="autumn")
    cbar = plt.colorbar(cs)
    # print('res2')
    # for i in res2:
    #     for j in i:
    #         print(j, end=' ')
    #     print('\n')
    # #ax.clabel(cs)
    plt.savefig('newton_res.png', dpi=300)
    ne_start = time.time()
    print("Время работы программы:", (ne_start - start)/60,'min')

if __name__ == "__main__":
    adv_5f()
    #base()
    #advanced()