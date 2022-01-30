# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:05:49 2020

@author: Skulpt-PC

Martin Buck, Math 123, Final project

Plot eigenvalues for different sigma
"""

import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt
import os

gauss_mat = loadmat('datasets/gaussian.mat')
gauss_data = gauss_mat['X']
ellipse_mat = loadmat('datasets/ellipses.mat')
ellipse_data = ellipse_mat['X']
circle_mat = loadmat('datasets/circles.mat')
circle_data = circle_mat['X']
spiral_mat = loadmat('datasets/spirals.mat')
spiral_data = spiral_mat['X']
moon_mat = loadmat('datasets/moon.mat')
moon_data = moon_mat['X']

s_arr = [.001, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 10, 100]


def main(data):
    """Plot eigenvalues for different values of sigma."""
    eigenvalues = np.zeros((np.shape(data)[0], len(s_arr)))
    eigenvectors = np.zeros((np.shape(data)[0], np.shape(data)[0], len(s_arr)))
    count = 0

    for s in s_arr:
        folder = 'plots/sigma = ' + str(s) + '/spirals'
        dataset_str = 'mat files/spiral_laplacian.mat_' + str(s) + '.mat'

        if not os.path.isdir(folder):
            os.makedirs(folder)

        if not os.path.isfile(dataset_str):
            L = laplacian(spiral_data, folder, dataset_str, s)
        else:
            laplacian_matrix = loadmat(dataset_str)
            L = laplacian_matrix['L']

        # Now compute eigenvectors of L
        w, v = np.linalg.eig(L)

        # get rid of imaginary parts
        if s < .1:
            w = np.real(w)
            v = np.real(v)

        # order and display eigenvalues and eigenvectors from smallest to
        # largest
        ind = np.argsort(w)

        w = w[ind]
        v = v[:, ind]

        eigenvalues[:, count] = w
        eigenvectors[:, :, count] = v

        plot_eigenvectors(v, w, folder, s)

        count += 1

    # plot eigenvalues and eigenvectors
    plot_eigenvalues(eigenvalues[:, 1:-2], folder)
    np.savetxt(folder + '/' + 'eigenvalues' + '.csv', eigenvalues,
               delimiter=',')


def laplacian(X, folder, dataset_str, s):
    """Create the Laplacian matrix."""
    n = np.shape(X)[0]
    W = np.zeros((n, n))
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            w = gauss(X[i, :], X[j, :], s)
            if i != j:
                W[i, j] = w
                W[j, i] = w
        D[i, i] = np.sum(W[i, :])
    # Laplacian is the difference of these two matrices
    L = D - W

    # visualize matrix
    plt.figure()
    plt.matshow(W)
    plt.title('Weight matrix')
    plt.savefig(folder + '/' + 'Weight matrix')
    plt.show()

    savemat(dataset_str, {'L': L})

    return L


def plot_eigenvalues(evals, folder):
    """Plot eigenvalues of Laplacian operator."""
    plt.figure()
    plt.plot(evals)
    plt.ylabel('Eigenvalues')
    plt.title('Eigenvalues of Laplacian')
    plt.savefig('Eigenvalues as a function of sigma.png')
    plt.show()


def plot_eigenvectors(evecs, evals, folder, s):
    """Plot eigenvectors of Laplacian operator."""
    handles = [221, 222, 223, 224]

    plt.figure()
    for i in range(4):
        plt.subplot(handles[i])
        plt.plot(evecs[:, i])
        plt.legend([str(round(evals[i], 4))])

    plt.savefig(folder + '/' + 'Eigenvectors')
    plt.show()


def gauss(x, y, sigma):
    """Calculate the Gaussian kernel."""
    return np.exp((-(np.linalg.norm(x-y)**2)/(2*(sigma**2))))


if __name__ == "__main__":
    main(spiral_data)
