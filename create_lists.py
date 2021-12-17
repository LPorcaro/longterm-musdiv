#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import csv

from operator import itemgetter
from itertools import combinations

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean, cdist

marker_types = [".", "o", "v", "^", "<", 
                ">", "1", "2", "3", "4",
                "8", "s", "p", "P", "h",
                "H", "+", "x", "X", "D",
                ".", "o", "v", "^", "<", '1']


METADATA = "data/filtered_tracks_20211124.csv"
METADATA_ENRICH = "data/filtered_tracks_enriched_20211124.csv"

LIST_DIV = "data/track_list_div.csv"
LIST_NOT_DIV = "data/track_list_not_div.csv"

MAP_GENRE = "data/map_genres.csv"
GENRE_DIST_MATRIX = "data/genres_distances.npy"
GENRE_INDEX = "data/genres_index.csv"


def import_embeddings(df):
    """
    """
    # Import embeddings
    embeddings = []
    
    for emb_path in df.emb_path:
        if os.path.exists(emb_path):
            embedding = np.load(emb_path)
            if len(embedding) != 200: # Sanity Check
                print(emb_path)
            else:
                embeddings.append(embedding)

    print("Found {} embeddings".format(len(embeddings)))
    return embeddings

def reduce_embeddings(embeddings):
    """
    """
    # PCA
    embeddings_stacked = np.vstack(embeddings)
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
    embeddings_reduced = projection.fit_transform(embeddings_reduced[:, :None])

    return embeddings_reduced

def get_centroid(embeddings_reduced):
    """
    """
    # Get centroid
    C_x = np.mean(emb_x)
    C_y = np.mean(emb_y)

    # Get Track max dist from Centroid
    dists = [euclidean(x, [C_x, C_y]) for x in embeddings_reduced]
    max_dist = np.max(dists)
    imax_dist = dists.index(max_dist)
    print("Max dist = {}".format(max_dist))

    return C_x, C_y, max_dist, imax_dist

def sort_tracks_by_distance(DistMatrix):
    """
    """
    # Sort tracks by average distance with 4 nn
    # Return list of tracks sorted with average distance
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

def create_lists(num_list, sort_avg_dists, df, diverse):
    """
    """
    print("Creating lists...")
    genres_allowed = ['techno', 'trance', 'hardcore', 'hardstyle']

    tracks_found = []
    genres_found = []
    nns = []


    for i in range(len(sort_avg_dists)):
        nn = DistMatrix[sort_avg_dists[i][0]].argsort()[:4]

        # Check track genre, get the most frequent
        list_genres = df_meta.iloc[nn].maingenre
        most_comm_genre = list_genres.value_counts().index.tolist()[0]

        # If there are more then two genres in the list, continue
        if len(list_genres.unique()) > 2:
            continue

        if diverse:
            if most_comm_genre in genres_found:
                continue
            else:
                genres_found.append(most_comm_genre)
        else:
            if most_comm_genre not in genres_allowed:
                continue
            else:
                genres_found.append(most_comm_genre)


        # Check if tracks already in other lists
        if not any(map(lambda v: v in tracks_found, nn)):
            nns.append(nn)
            tracks_found.extend(nn)

        if len(nns) == num_list:
            break

    # for i in range(k):
    #     print("List {} ({})".format(nns[i], genres_found[i]))
    #     print("{}".format([df.iloc[c].yt_id for c in nns[i]]))

    # Write track lists
    if diverse:
        outfile = LIST_DIV
    else:
        outfile = LIST_NOT_DIV

    with open(outfile, 'w+') as outf:
        _writer = csv.writer(outf)
        for i in range(num_list):
            _writer.writerow([df.iloc[c].yt_id for c in nns[i]])


    return nns, genres_found


if __name__ == "__main__":

    num_list = 14

    df_meta = pd.read_csv(METADATA_ENRICH)
    embeddings = import_embeddings(df_meta)
    embeddings_red = reduce_embeddings(embeddings)

    emb_x = list(map(itemgetter(0), embeddings_red))
    emb_y = list(map(itemgetter(1), embeddings_red))

    C_x, C_y, max_dist, imax_dist = get_centroid(embeddings_red)

    DistMatrix = cdist(embeddings_red, embeddings_red, 'minkowski')
    sort_avg_dists = sort_tracks_by_distance(DistMatrix)
    nns_div, genres_found_div = create_lists(num_list, sort_avg_dists, df_meta, diverse=True)
    nns, genres_found = create_lists(num_list, sort_avg_dists, df_meta, diverse=False)


    # Plot
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.scatter(C_x, C_y)
    ax1.set_title('List non diversified')
    for c in range(num_list):
        ax1.scatter([emb_x[i] for i in nns[c]], [emb_y[i] for i in nns[c]])
        ax1.annotate(df_meta.iloc[nns[c][0]].maingenre, (emb_x[nns[c][0]], emb_y[nns[c][0]]))

    ax2.scatter(C_x, C_y)
    ax2.set_title('List diversified')
    for c in range(num_list):
        ax2.scatter([emb_x[i] for i in nns_div[c]], [emb_y[i] for i in nns_div[c]])
        ax2.annotate(df_meta.iloc[nns_div[c][0]].maingenre, (emb_x[nns_div[c][0]], emb_y[nns_div[c][0]]))


    ax1.annotate('Centroid', (C_x, C_y))
    cir = plt.Circle((C_x, C_y), max_dist, color='r',fill=False)
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.add_patch(cir)
    cir = plt.Circle((C_x, C_y), max_dist, color='r',fill=False)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.add_patch(cir)
    plt.show()