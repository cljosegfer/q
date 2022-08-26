#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:17:04 2022

@author: jose
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.datasets import make_circles

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
sd = 0.01
n = 200
X, y = make_circles(n_samples = n, noise = sd, factor = 0.5, shuffle = False)

# gg
gg = lara_graph(X)
scores = q_index(X, y, gg)

# classifier
distances = []
erro = []
for i, x in enumerate(X):
    raio = np.linalg.norm(x)
    if i < n//2:
        e = 0 if raio > 0.75 else 1
    else:
        e = 0 if raio <= 0.75 else 1
    d = np.min([np.abs(x[0]), np.abs(x[1])])
    if e:
        d *= -1
    distances.append(d)
    erro.append(e)
distances = np.array(distances)
erro = np.array(erro)

# plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = y)
border = plt.Circle((0, 0), 0.75, color='blue', fill = False)
plt.gca().add_patch(border)
plt.title('sd = {}, q_mean = {}'.format(sd, np.mean(scores)))
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
