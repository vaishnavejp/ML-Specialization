import numpy as np
import matplotlib.pyplot as plt
import math


def load_data():
    data = np.loadtxt("winequal.csv", delimiter=",", skiprows=1)
    X = data[:1000, :11]
    y = data[:1000, 11]
    return X, y


def analyze_dataset():
    x, y = load_data()
    print("First five elements of the dataset : ", x[:6, :])
    print("Datatype of the elements :", type(x), type(y))
    print("Dimensions of the data : x & y", x.shape, y.shape)


def cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost


def gradient_cost(x, y, w, b):
    m = x.shape[0]
    n = x.shape[1]

    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        f_wb = np.dot(w, x[i]) + b
        for j in range(n):
            dj_dw += (f_wb - y[i]) * x[i]
        dj_db += f_wb - y[i]

    dj_db /= m
    dj_dw /= m

    return dj_dw, dj_db


def gradient_descent(x, y, w, b, alpha, num_iter):
    J_hist = []
    for i in range(num_iter):
        dj_dw, dj_db = gradient_cost(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_hist.append(cost(x, y, w, b))

        if i % math.ceil(num_iter / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_hist[-1]:8.2f}   ")

    return w, b, J_hist


def init_params(x, y):
    initial_w = np.zeros((x.shape[1],))
    initial_b = 0

    iter = 1000
    alpha = 7.0e-7

    w_final, b_final, J_hist = gradient_descent(x, y, initial_w, initial_b, alpha, iter)

    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    return w_final, b_final


def predict(w, b):
    data = np.loadtxt("winequal.csv", delimiter=",", skiprows=1)
    X = data[1001:1600, :11]
    y = data[1001:1600, 11]

    for i in range(X.shape[0]):
        pred_val = np.dot(w, X[i]) + b

        if i > 300 and i < 310:
            print("predicted = ", pred_val, "; actual = ", y[i])


x, y = load_data()
w, b = init_params(x, y)
predict(w, b)
