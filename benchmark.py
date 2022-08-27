#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 19:57:48 2022

@author: jose
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy import io
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

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

# params
dir_path = 'data'
datasets = ('australian', 
            'banknote', 
            'breastcancer', 
            'breastHess', 
            'bupa', 
            'climate', 
            'diabetes', 
            'fertility', 
            'german', 
            'golub', 
            'haberman', 
            'heart', 
            'ILPD', 
            'parkinsons', 
            'sonar')
K = 10

dataset = 'australian'
fold_n = 0

# read
filename = '{}/exportBase_{}_folds_10_exec_{}.mat'.format(
    dir_path, dataset, fold_n + 1)
data_mat = io.loadmat(filename)

# train / test
train = data_mat['data']['train'][0][0]
classTrain = data_mat['data']['classTrain'][0][0].ravel()
test = data_mat['data']['test'][0][0]
classTest = data_mat['data']['classTest'][0][0].ravel()

# data
X = np.concatenate((train, test), axis = 0)
y = np.concatenate((classTrain, classTest), axis = 0)

# gg
gg = lara_graph(X)
scores = q_index(X, y, gg)

# clc
model = SVC()
model.fit(train, classTrain)

acc = roc_auc_score(classTest, model.predict(test))
yhat = model.predict(X)
erro = y != yhat

distances = model.decision_function(X)
distances[y == -1] *= -1

# plot
print(acc, np.mean(scores))
log = np.stack((scores, distances)).T
colors = {0:'blue', 1:'red'}

plt.figure()
plt.scatter(log[:, 0], log[:, 1], c = [colors[e] for e in erro[:]])
plt.axhline(y = 0)
plt.axvline(x = np.mean(scores), color="green", linestyle="--")
plt.xlim([0, 1])
plt.title('{}, f(x) vs q, auc = {:.3f}'.format(dataset, acc))
plt.xlabel('q')
plt.ylabel('f(x)')

plt.figure()
y_lower = 10
for c in [1, 0]:
    cluster = erro == c
    ith_cluster_silhouette_values = scores[cluster]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

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
plt.axvline(x=np.mean(scores), color="green", linestyle="--")
plt.yticks([])
plt.title('{}, q_mean = {:.3f}'.format(dataset, np.mean(scores)))
plt.xlabel('q')
