import numpy as np
import matplotlib.pyplot as plt


def create_plot(n):
    figure = plt.figure(figsize=(6, 6))
    ax = figure.add_subplot()
    ax.set_autoscaley_on(True)
    ax.set_xlim(-1, n + 1)
    ax.set_ylim(-1, n + 1)
    return ax


def plotter(ax, v):
    plt.cla()
    ax.axis('off')
    ax.set_autoscaley_on(True)
    plt.matshow(v, fignum=0)
    plt.draw()
    plt.show()
    plt.pause(0.1)


def plotter_policy(ax, pi):
    plt.cla()
    ax.axis('off')
    ax.set_autoscaley_on(True)
    n, _, _ = np.shape(pi)
    X = range(0, n)
    Y = range(0, n)
    scale_factor = 0.01
    scaled_pi = scale_factor * pi
    V = scaled_pi[:, :, 0]
    U = np.zeros_like(V)
    plt.quiver(X, Y, U, V)

    V = -1. * scaled_pi[:, :, 1]
    U = np.zeros_like(V)
    plt.quiver(X, Y, U, V)

    U = scaled_pi[:, :, 2]
    V = np.zeros_like(U)
    plt.quiver(X, Y, U, V)

    U = -1. * scaled_pi[:, :, 3]
    V = np.zeros_like(U)
    plt.quiver(X, Y, U, V)

    plt.draw()
    plt.show()
    plt.pause(0.4)
