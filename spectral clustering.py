# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 09:44:29 2020

@author: Skulpt-PC

Martin Buck, Math 123 project

Implement spectral clustering
"""

import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import random
import time
import os

# load simple toy data sets to implement spectral clustering on. This includes
# two gaussian blobs, two ellipse blobs, the "moon" data set, and two
# concentric circles
spiral_mat = loadmat('datasets/spirals.mat')
spiral_data = spiral_mat['X']

# epsilon determines when k-means should stop iterating
epsilon = 10**-18
# need to tune parameter sigma
s_arr = [.001, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 10, 100]
# s_arr = [.001]
# clusters
k = 2
# number of eigenvectors to explore embedding
e_start = 0
e_end = 3

start_time = time.time()


def main():
    """Run the spectral clustering alg. on the above data sets."""
    eigen_errors = np.zeros((len(s_arr),))
    error_count = 0
    for s in s_arr:
        folder = 'plots/sigma = ' + str(s) + '/spirals'
        if not os.path.isdir(folder):
            os.makedirs(folder)
        eigen_errors[error_count] = \
            spect(spiral_data, 'mat files/spiral_laplacian.mat_' + str(s) +
                  '.mat', folder, s)
        error_count += 1

    # plot error as a function of number of eigenvectors included, and error
    # as a function of sigma. Create surface plot
    plt.plot(eigen_errors)


def spect(X, dataset_str, folder, s):
    """Cluster data using eigenvectors of Laplacian and k-means."""
    # First center and scale data and plot
    X = scale(X)
    plot_data(X, folder)

    if not os.path.isfile(dataset_str):
        L = laplacian(X, folder, dataset_str, s)
    else:
        laplacian_matrix = loadmat(dataset_str)
        L = laplacian_matrix['L']

    # Now compute eigenvectors of L
    w, v = np.linalg.eig(L)
    # order and display eigenvalues and eigenvectors from smallest to largest
    ind = np.argsort(w)

    # get rid of imaginary parts
    if s < .1:
        w = np.real(w)
        v = np.real(v)

    w = w[ind]
    v = v[:, ind]

    # run k-means on embedded data
    # first iteration looks at just the 2nd eigenvector which may or may not
    # be the fiedler vector
    # probably need to identify the first non-constant eigenvector
    phi_X = v[:, 2:4]
    eigen_error = k_means(phi_X, X, 2, folder, s)

    return eigen_error


def k_means(X, X_o, e, folder, s):
    """Run k-means on data set X."""
    # randomly assign centers. Assume two clusters for now
    mu1 = [random.uniform(-.005, .005) for i in range(e)]
    mu2 = [random.uniform(-.005, .005) for i in range(e)]

    # will store sum of squared 2-norm errors
    errors = np.zeros((0,))
    error_count = 1

    # run k-means twice in order to initialize error condition in while loop
    labels, error = label(X, mu1, mu2)
    errors = np.append(errors, error)
    mu1, mu2 = find_centers(X, labels, e)
    # plot labeled embedded data and labeled orginal data
    if e < 4:
        plot_clusters(X, labels, s, e, 1, error_count, folder)
    plot_clusters(X_o, labels, s, e, 0, error_count, folder)
    error_count += 1

    labels, error = label(X, mu1, mu2)
    errors = np.append(errors, error)
    mu1, mu2 = find_centers(X, labels, e)
    # plot labeled embedded data and labeled orginal data
    if e < 4:
        plot_clusters(X, labels, s, e, 1, error_count, folder)
    plot_clusters(X_o, labels, s, e, 0, error_count, folder)
    error_count += 1

    for i in range(20):
        labels, error = label(X, mu1, mu2)
        errors = np.append(errors, error)
        mu1, mu2 = find_centers(X, labels, e)
        # plot labeled embedded data and labeled original data
        if e < 4:
            plot_clusters(X, labels, s, e, 1, error_count, folder)
        plot_clusters(X_o, labels, s, e, 0, error_count, folder)
        error_count += 1

    plot_errors(errors, error_count-1, e, folder)
    plt.title('Errors vs. Sigma')
    plt.xlabel('Sigma')
    plt.ylabel('Square Euclidean Error')
    plt.xticks(range(14), ['.001', '.01', '.1', '.2', '.3', '.4', '.5', '.6'
                           '.7', '.8', '.9', '1', '10', '100'])

    return errors[-1]


def laplacian(X, folder, dataset_str, s):
    """Create the Laplacian matrix."""
    n = np.shape(X)[0]
    W = np.zeros((n, n))
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            w = gauss(X[i, :], X[j, :], s)
            # w = inner(X[i, :], X[j, :], d)
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


def label(X, mu1, mu2):
    """Label each data point by its closest center and calculate error."""
    # now iterate over all the points in the data set and calculate the
    # distance to each center and assign a label to the closer center
    n = np.shape(X)[0]
    labels = np.zeros((n,))
    error = 0
    for i in range(n):
        x = X[i, :]
        d1 = euclidean(x, mu1)**2
        d2 = euclidean(x, mu2)**2
        if d1 < d2:
            labels[i] = 1
            error += d1
        else:
            labels[i] = 2
            error += d2

    return labels, error


def find_centers(X, labels, e):
    """Find new centers of clusters."""
    # now update the means of each cluster
    n = np.shape(X)[0]
    sum1 = [0 for i in range(e)]
    sum2 = [0 for i in range(e)]
    count1 = 0
    count2 = 0
    for i in range(n):
        x = X[i, :]
        if labels[i] == 1:
            sum1 += x
            count1 += 1
        else:
            sum2 += x
            count2 += 1

    if count1 > 0:
        mu1 = sum1/count1
    else:
        mu1 = [random.uniform(-.005, .005) for i in range(e)]

    if count2 > 0:
        mu2 = sum2/count2
    else:
        mu2 = [random.uniform(-.005, .005) for i in range(e)]

    return mu1, mu2


def plot_data(X, folder):
    """Plot scaled data to get a feel for what it looks like."""
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.title('Standardized data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(folder + '/' + 'Standardized data')
    plt.show()


def plot_clusters(X, labels, s, e, eigen, error_count, folder):
    """Plot and color code clusters."""
    # plot original 2-D clusters with labels from k-means on embedding
    if not eigen:
        plt.scatter(X[labels == 1, 0], X[labels == 1, 1], c='r')
        plt.scatter(X[labels == 2, 0], X[labels == 2, 1], c='b')
        plt.title('sigma = ' + str(s) + ', ' + str(e) +
                  ' eigenvector(s), ' +
                  'iteration ' + str(error_count))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.savefig(folder + '/' + str(e) +
                    ' eigenvector(s), ' +
                    'iteration ' + str(error_count))
        plt.show()
    # if only using 1 eigenvector, plot embedded data that k-means was
    # performed on
    elif eigen and len(np.shape(X)) == 1 or np.shape(X)[1] == 1:
        y1 = np.zeros((len(X[labels == 1, 0]),))
        y2 = np.zeros((len(X[labels == 2, 0]),))
        plt.scatter(X[labels == 1, 0], y1, c='r')
        plt.scatter(X[labels == 2, 0], y2, c='b')
        plt.savefig(folder + '/' + str(e) +
                    ' embedded data, ' +
                    'iteration ' + str(error_count))
        plt.show()
    # if only using 2 eigenvectors, plot embedded data that k-means was
    # performed on
    elif eigen and np.shape(X)[1] == 2:
        plt.scatter(X[labels == 1, 0], X[labels == 1, 1], c='r')
        plt.scatter(X[labels == 2, 0], X[labels == 2, 1], c='b')
        plt.title('sigma = ' + str(s) + ', ' + str(e) +
                  'eigenvector(s) embedded data, ' +
                  'iteration ' + str(error_count))
        plt.savefig(folder + '/' + str(e) +
                    'eigenvector(s) embedded data, ' +
                    'iteration ' + str(error_count))
        plt.show()
    # if only using 3 eigenvectors, plot embedded data that k-means was
    # performed on
    elif eigen and np.shape(X)[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[labels == 1, 0], X[labels == 1, 1], X[labels == 1, 2],
                   c='r')
        ax.scatter(X[labels == 2, 0], X[labels == 2, 1], X[labels == 2, 2],
                   c='b')
        plt.savefig(folder + '/' + str(e) +
                    'eigenvector(s) embedded data, ' +
                    'iteration ' + str(error_count))
        plt.show()


def plot_errors(errors, error_count, e, folder):
    """Plot k-means error as a function of number of iterations."""
    plt.plot(range(error_count), errors)
    plt.xlabel('Iterations')
    plt.ylabel('Squared euclidean error')
    plt.title('K-means error vs. iterations')
    plt.savefig(folder + '/' + 'K-means error vs iterations, ' + str(e) +
                ' eigenvectors')
    plt.show()


def gauss(x, y, sigma):
    """Calculate the Gaussian kernel."""
    return np.exp((-(np.linalg.norm(x-y)**2)/(2*(sigma**2))))


def inner(x, y, d):
    """Calculate the power of dot product kernel."""
    return np.dot(x, y)**d


def scale(X):
    """Center and scale data so it is mean zero and unit variance."""
    return (X - np.mean(X, axis=0))/np.std(X, axis=0)


if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))
