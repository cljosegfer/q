#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 08:51:03 2022

@author: jose
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def lara_graph(X):
    D = distance_matrix(X, X) ** 2
    D[np.diag_indices(D.shape[0])] = 1e6

    n = X.shape[0]
    adjacencia = np.zeros(shape = (n, n))
    for i in tqdm(range(n-1)):
        for j in range(i+1, n):
            minimo = min(D[i, :] + D[j, :])
            if (D[i, j] <= minimo):
                adjacencia[i, j] = 1
                adjacencia[j, i] = 1
    return adjacencia

def q_index(X, y, gg):
    scores = []
    for i, row in enumerate(gg):
        vizinhos = np.where(row == 1)[0]
        
        degree = len(vizinhos)
        opposite = 0
        for vizinho in vizinhos:
            opposite += np.exp(-np.linalg.norm(X[i] - X[vizinho])) * np.abs(y[i] - y[vizinho]) / 2
        q = 1 - opposite / degree
        scores.append(q)
    scores = np.array(scores)
    return scores

# data
sd = 0.5
n = 100

mean = (0, 0)
cov = [[sd, 0], [0, sd]]
x1 = np.random.multivariate_normal(mean, cov, size = n)

mean = (1, 1)
cov = [[sd, 0], [0, sd]]
x2 = np.random.multivariate_normal(mean, cov, size = n)

X = np.concatenate((x1, x2))
y = np.array([-1] * n + [1] * n)

# gg
gg = lara_graph(X)

# scores = []
# for i, row in enumerate(gg):
#     vizinhos = np.where(row == 1)[0]
    
#     degree = len(vizinhos)
#     opposite = 0
#     for vizinho in vizinhos:
#         opposite += np.exp(-np.linalg.norm(X[i] - X[vizinho])) * np.abs(y[i] - y[vizinho]) / 2
#     q = 1 - opposite / degree
#     scores.append(q)
# scores = np.array(scores)
scores = q_index(X, y, gg)
print(sd, np.mean(scores))

# classifier
w = [-1, 1, 1]
w = w / np.linalg.norm(w)

distances = []
for i, x in enumerate(X):
    x = np.hstack((1, x))
    # d = np.dot(x, w)
    d = np.dot(x, w) * np.sign(y[i])
    distances.append(d)
distances = np.array(distances)
# erro = (np.sign(distances) == np.sign(y)) * 1
erro = distances < 0

# plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.axline((0, 1), (1, 0))
plt.title('sd = {}, q_mean = {}'.format(sd, np.mean(scores)))
plt.xlim([-1, 2])
plt.ylim([-1, 2])

log = np.stack((scores, distances)).T
colors = {0:'blue', 1:'red'}

plt.figure()
plt.scatter(log[:, 0], log[:, 1], c = [colors[e] for e in erro[:]])
plt.axhline(y=0)
plt.xlim([0, 1])
plt.title('<x, w> vs q, sd = {}'.format(sd))
plt.xlabel('q')
plt.ylabel('<x, w>')

plt.figure()
y_lower = 10
for c in [1, 0]:
    cluster = erro == c
    ith_cluster_silhouette_values = scores[cluster]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    # color = cm.nipy_spectral(float(c) / len(np.unique(y)))
    color = colors[c]
    plt.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )
    y_lower = y_upper + 10
plt.axvline(x=np.mean(scores), color="red", linestyle="--")
plt.yticks([])
