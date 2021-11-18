#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

from operator import itemgetter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, euclidean

EMB_DIR = "/home/lorenzo/Data/test_musicexplore/data/msd-musicnn-embeddings"


# Import embeddings
embeddings = []
fnames = []
for file in sorted(os.listdir(EMB_DIR)):
    fname, exts = os.path.splitext(file)
    if exts == '.npy':
        embeddings.append(np.load(os.path.join(EMB_DIR, file)))
        fnames.append(fname)


# PCA
embeddings_stacked = np.vstack(embeddings)
lengths = list(map(len, embeddings))
projection = PCA(random_state=0, copy=False)
projection = projection.fit(embeddings_stacked[:, :None])

threshold = 0.8
pc_num = 1
exp_var_ratio = 0
while exp_var_ratio <= threshold:
    exp_var_ratio = np.sum(projection.explained_variance_ratio_[:pc_num])
    pc_num += 1

print ("Explained variance ratio by {} PC: {}".format(pc_num, exp_var_ratio))

projection = PCA(random_state=0, copy=False, n_components=pc_num)
embeddings_reduced = projection.fit_transform(embeddings_stacked[:, :None])


# TSNE
projection = TSNE(n_components=2, perplexity=5, random_state=1, n_iter=500, init='pca', verbose=True)
lengths = list(map(len, embeddings_reduced))
embeddings_reduced = projection.fit_transform(embeddings_reduced[:, :None])


x = list(map(itemgetter(0), embeddings_reduced))
y = list(map(itemgetter(1), embeddings_reduced))


# Get centroid
C_x = np.mean(x)
C_y = np.mean(y)

# Get max dist
dists = [euclidean(x, [C_x, C_y]) for x in embeddings_reduced]
max_dist = np.max(dists)
imax_dist = dists.index(max_dist)
fnames[imax_dist] = fnames[imax_dist]+ '- MAX'


# Plot
fig, ax = plt.subplots()
ax.scatter(x, y)
for i, label in enumerate(fnames):
    plt.annotate(label, (x[i], y[i]))
ax.scatter(C_x, C_y)
plt.annotate('Centroid', (C_x, C_y))
cir = plt.Circle((C_x, C_y), max_dist, color='r',fill=False)
ax.set_aspect('equal', adjustable='datalim')
ax.add_patch(cir)
plt.show()

