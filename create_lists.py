#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json 
import pandas as pd
import seaborn as sns 
import mantel

from tqdm import tqdm
from collections import OrderedDict
from operator import itemgetter
from itertools import combinations

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean, cdist
from sklearn.metrics import silhouette_score, silhouette_samples

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import PercentFormatter

marker_types = [".", "o", "v", "^", "<", 
                ">", "1", "2", "3", "4",
                "8", "s", "p", "P", "h",
                "H", "+", "x", "X", "D",
                ".", "o", "v", "^", "<", '1']


EMB_DIR = "/home/lorenzo/Data/longterm_data/effnet_filtered_20211124/"
ESS_DIR = "/home/lorenzo/Data/longterm_data/essentia_20211124/"

METADATA = "data/filtered_tracks_20211124.csv"
MAP_GENRE = "data/map_genres.csv"
GENRE_DIST_MATRIX = "data/genres_distances.npy"
GENRE_INDEX = "data/genres_index.csv"



def import_metadata(meta_file):
    """
    """
    df = pd.read_csv(meta_file)
    df_map_genre = pd.read_csv(MAP_GENRE)
    df_genre_index = pd.read_csv(GENRE_INDEX)

    df['yt_id'] = [x.split('?v=')[1] for x in df['yt_link']]
    df_new = pd.merge(df, df_map_genre, on='genre')
    df_new = df_new.sort_values(by=['maingenre','genre'])

    # Remove tracks with no embedding
    for t_id in df_new.yt_id:
        file = t_id + '.npy'
        file_path = os.path.join(EMB_DIR, file)
        if not os.path.exists(file_path):
            df_new = df_new.drop(df_new[df_new.yt_id ==t_id].index.values)

    genres = sorted(df_new['maingenre'].unique())

    genres_dist_matrix = np.load(GENRE_DIST_MATRIX)
    df_new = df_new.drop_duplicates(subset=('yt_id'), keep='last')
    df_new = df_new[df_new['maingenre'] != 'rock']

    return df_new, df_genre_index, df_map_genre, genres_dist_matrix, genres


def import_embeddings(emb_dir):
    """
    """
    # Import embeddings
    embeddings = []
    fnames = []
    
    for t_id in df_meta.yt_id:
        file = t_id + '.npy'
        file_path = os.path.join(emb_dir, file)
        if os.path.exists(file_path):
            embedding = np.load(file_path)
            if len(embedding) != 200: # Sanity Check
                print(fname)
            else:
                embeddings.append(embedding)
                fnames.append(t_id)

    return embeddings, fnames


def reduce_embeddings(embeddings):
    """
    """
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
    projection = TSNE(n_components=2, perplexity=7, random_state=1, n_iter=500, init='pca', verbose=True)
    lengths = list(map(len, embeddings_reduced))
    embeddings_reduced = projection.fit_transform(embeddings_reduced[:, :None])

    emb_x = list(map(itemgetter(0), embeddings_reduced))
    emb_y = list(map(itemgetter(1), embeddings_reduced))

    return embeddings_reduced, emb_x, emb_y


def get_centroid(embeddings_reduced, fnames):
    """
    """
    # Get centroid
    C_x = np.mean(emb_x)
    C_y = np.mean(emb_y)

    # Get Track max dist from Centroid
    dists = [euclidean(x, [C_x, C_y]) for x in embeddings_reduced]
    max_dist = np.max(dists)
    imax_dist = dists.index(max_dist)
    # fnames[imax_dist] = fnames[imax_dist]+ '- MAX'
    print("Max dist = {}".format(max_dist))

    return C_x, C_y, max_dist

def sort_tracks_by_distance(DistMatrix):
    """
    """
    # Sort nn by average distances
    avg_dists = []
    for i in range(len(DistMatrix)):
        nn_dists = []
        nn = DistMatrix[i].argsort()[:4]
        comb_nn = combinations(nn, 2)
        for n1, n2 in comb_nn:
            nn_dists.append(DistMatrix[n1,n2])

        avg_dists.append((i, np.average(nn_dists)))

    sort_avg_dists = sorted(avg_dists, key = lambda x: x[1])

    return sort_avg_dists

def create_lists(sort_avg_dists):
    """
    """
    num_list = 4
    k = 0 
    tracks_found = []
    nns = []
    for i in range(len(sort_avg_dists)):
        nn = DistMatrix[sort_avg_dists[i][0]].argsort()[:4]
        if not any(map(lambda v: v in tracks_found, nn)):
            nns.append(nn)
            tracks_found.extend(nn)
            k += 1
            if k == num_list:
                break

    print("List {}".format(nns[0]))
    print("{}".format([fnames[i] for i in nns[0]]))
    print("List {}".format(nns[1]))
    print("{}".format([fnames[i] for i in nns[1]]))
    print("List {}".format(nns[2]))
    print("{}".format([fnames[i] for i in nns[2]]))
    print("List {}".format(nns[3]))
    print("{}".format([fnames[i] for i in nns[3]]))

    return nns


if __name__ == "__main__":


    df_meta, df_genre_index, df_map_genre, DistMatrixGenre, genres = import_metadata(METADATA)
    embeddings, fnames = import_embeddings(EMB_DIR)
    embeddings_reduced, emb_x, emb_y = reduce_embeddings(embeddings)
    C_x, C_y, max_dist = get_centroid(embeddings_reduced, fnames)
    DistMatrix = cdist(embeddings_reduced, embeddings_reduced, 'minkowski')
    sort_avg_dists = sort_tracks_by_distance(DistMatrix)
    nns = create_lists(sort_avg_dists)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(emb_x, emb_y)
    for i, label in enumerate(fnames):
        if any(i in subl for subl in  nns):
            plt.annotate(label, (emb_x[i], emb_y[i]))
    ax.scatter(C_x, C_y)
    ax.scatter([emb_x[i] for i in nns[0]], [emb_y[i] for i in nns[0]])
    ax.scatter([emb_x[i] for i in nns[1]], [emb_y[i] for i in nns[1]])
    ax.scatter([emb_x[i] for i in nns[2]], [emb_y[i] for i in nns[2]])
    ax.scatter([emb_x[i] for i in nns[3]], [emb_y[i] for i in nns[3]])

    plt.annotate('Centroid', (C_x, C_y))
    cir = plt.Circle((C_x, C_y), max_dist, color='r',fill=False)
    ax.set_aspect('equal', adjustable='datalim')
    ax.add_patch(cir)
    plt.show()