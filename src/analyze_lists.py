#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import csv
import numpy as np
import os
import matplotlib.pyplot as plt

from itertools import combinations
from tabulate import tabulate
from scipy.spatial.distance import cdist, euclidean, cosine
from operator import itemgetter
from scipy import stats
from mannwhitney import mannWhitney


marker_types = [".", "o", "v", "^", "<",
                ">", "1", "2", "3", "4",
                "8", "s", "p", "P", "h",
                "H", "+", "x", "X", "D",
                ".", "o", "v", "^", "<", '1']

np.random.shuffle(marker_types)


ESSENTIA_DIR = "/home/lorenzo/Data/longterm_data/features/"
EMB_DIR = "../data/embeddings/{}"
TRACKS = "../data/input/random_tracklist_20220104.csv"
TRACKS_FEAT = "../data/input/tracklist_features_20220104.csv"

CREATION_TIME = "20220122_140806"
LIST_DIV = "../data/lists/track_list_div_{}.csv".format(CREATION_TIME)
LIST_NOT_DIV = "../data/lists/track_list_not_div_{}.csv".format(CREATION_TIME)


def import_embeddings(emb_dir, emb_type, emb_length):
    """
    """
    embeddings = []
    filenames = []

    emb_dir_input = emb_dir.format(emb_type)
    for emb_file in os.listdir(emb_dir_input):
        if not emb_file.endswith('.npy'):
            continue

        filename, ext = os.path.splitext(emb_file)

        file_path = os.path.join(emb_dir_input, emb_file)
        if os.path.exists(file_path):
            embedding = np.load(file_path)
            if len(embedding) != emb_length:
                print(filename)
            else:
                embeddings.append(embedding)
                filenames.append(filename)

    return embeddings, filenames


def import_lists():
    """
    """

    list_div, list_not_div = [], []
    list_div_genres, list_not_div_genres = [], []

    with open(LIST_DIV, 'r') as inf1, open(LIST_NOT_DIV, 'r') as inf2:
        _reader1 = csv.reader(inf1)
        _reader2 = csv.reader(inf2)

        for row in _reader1:
            list_div.append(row[:-1])
            list_div_genres.append(row[-1])

        for row in _reader2:
            list_not_div.append(row[:-1])
            list_not_div_genres.append(row[-1])

    return list_div, list_not_div, list_div_genres, list_not_div_genres


def find_indexes(list_div, list_not_div, filenames):
    """
    """
    # Get nested list of indexes
    list_div_idxs = []
    for l in list_div:
        list_div_idxs.append([filenames.index(track) for track in l])

    list_not_div_idxs = []
    for l in list_not_div:
        list_not_div_idxs.append([filenames.index(track) for track in l])

    return list_div_idxs, list_not_div_idxs


def get_centroid(emb_x, emb_y, embeddings):
    """
    """
    # Get centroid
    C_x = np.mean(emb_x)
    C_y = np.mean(emb_y)

    # Get Track max dist from Centroid
    dists = [euclidean(x, [C_x, C_y]) for x in embeddings]
    max_dist = np.max(dists)
    imax_dist = dists.index(max_dist)
    # print("Max dist = {}".format(max_dist))

    return C_x, C_y, max_dist, imax_dist


def plot_lists(embeddings, nns_div, nns, nns_div_genres, nns_genres):
    """
    """
    num_list = len(nns_div_genres)

    emb_x = list(map(itemgetter(0), embeddings))
    emb_y = list(map(itemgetter(1), embeddings))
    C_x, C_y, max_dist, imax_dist = get_centroid(emb_x, emb_y, embeddings)

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.scatter(C_x, C_y)
    ax1.set_title('Low Diversity Lists')
    for c in range(num_list):
        x = np.mean([emb_x[i] for i in nns[c]])
        y = np.mean([emb_y[i] for i in nns[c]])
        ax1.scatter(x, y, marker=marker_types[c], label=nns_div_genres[c])
        text = ax1.annotate(nns_genres[c], (x, y))
        text.set_fontsize(15)

    ax2.scatter(C_x, C_y)
    ax2.set_title('High Diversity Lists')
    for c in range(num_list):
        x = np.mean([emb_x[i] for i in nns_div[c]])
        y = np.mean([emb_y[i] for i in nns_div[c]])
        ax2.scatter(x, y, marker=marker_types[c], label=nns_div_genres[c])
        text = ax2.annotate(nns_div_genres[c], (x, y))
        text.set_fontsize(15)


    ax1.annotate('Centroid', (C_x, C_y))
    cir = plt.Circle((C_x, C_y), max_dist, color='r', fill=False)
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.add_patch(cir)
    ax2.annotate('Centroid', (C_x, C_y))
    cir = plt.Circle((C_x, C_y), max_dist, color='r', fill=False)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.add_patch(cir)

    plt.show()


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def test_significance(list1, list2):
    """
    """
    MU = mannWhitney(list1, list2)
    print("Medians Diff Distances -> {:.3f} - {:.3f}".format(
        np.median(list1), np.median(list2)))
    print("Mann Whitney -> Significance: {};"
          " U-statistics: {}, EffectSize: {:.3f}".format(
            MU.significance, MU.u, MU.effectsize))

    tStat, pValue = stats.ttest_ind(list1, list2)
    print("T-Test -> P-Value:{} T-Statistic:{:.3f}".format(pValue, tStat))


def test_distances():
    """
    """
    # Get flat list index
    div_idxs = [item for elem in list_div_idxs for item in elem]
    not_div_idxs = [item for elem in list_not_div_idxs for item in elem]

    l_embs = ["effnet", "effnet_pca", "effnet_tsne"]
    s_embs = [200, 17, 2]

    for l_emb, s_embs in zip(l_embs, s_embs):

        embeddings, filenames = import_embeddings(EMB_DIR, l_emb, s_embs)
        embeddings = np.vstack(embeddings)

        print("\n########### {} EMBEDDINGS".format(l_emb))
        print("#### Distance from geometric median")
        div_gmedian = geometric_median(
            np.vstack(itemgetter(*div_idxs)(embeddings)))
        div_dists = []
        for idx in div_idxs:
            dist = cosine(div_gmedian, embeddings[idx])
            div_dists.append(dist)

        not_div_gmedian = geometric_median(
            np.vstack(itemgetter(*not_div_idxs)(embeddings)))
        not_div_dists = []
        for idx in not_div_idxs:
            dist = cosine(not_div_gmedian, embeddings[idx])
            not_div_dists.append(dist)

        test_significance(div_dists, not_div_dists)

        print("#### Inter-list average distance")
        avg_inter_list_1 = []
        for c1, l1 in enumerate(list_div_idxs):
            for c2, l2 in enumerate(list_div_idxs):
                if c1 <= c2:
                    continue
                emb1 = itemgetter(*l1)(embeddings)
                emb2 = itemgetter(*l2)(embeddings)
                distances = cdist(emb1, emb2, 'cosine')
                avg_dist = np.average(
                    distances[np.nonzero(np.triu(distances))])
                avg_inter_list_1.append(avg_dist)

        avg_inter_list_2 = []
        for c1, l1 in enumerate(list_not_div_idxs):
            for c2, l2 in enumerate(list_not_div_idxs):
                if c1 <= c2:
                    continue
                emb1 = itemgetter(*l1)(embeddings)
                emb2 = itemgetter(*l2)(embeddings)
                distances = cdist(emb1, emb2, 'cosine')
                avg_dist = np.average(
                    distances[np.nonzero(np.triu(distances))])
                avg_inter_list_2.append(avg_dist)

        test_significance(avg_inter_list_1, avg_inter_list_2)

        print("#### Intra-list average distance")
        avg_intra_list_1 = []
        for idxs in list_div_idxs:
            emb = itemgetter(*idxs)(embeddings)
            distances = cdist(emb, emb, 'cosine')
            avg_dist = np.average(distances[np.nonzero(np.triu(distances))])
            avg_intra_list_1.append(avg_dist)

        avg_intra_list_2 = []
        for idxs in list_not_div_idxs:
            emb = itemgetter(*idxs)(embeddings)
            distances = cdist(emb, emb, 'cosine')
            avg_dist = np.average(distances[np.nonzero(np.triu(distances))])
            avg_intra_list_2.append(avg_dist)

        test_significance(avg_intra_list_1, avg_intra_list_2)


def test_features():
    """
    """
    df_sp_feat = pd.read_csv(TRACKS_FEAT, delimiter='\t')
    df_sp_feat = df_sp_feat.drop_duplicates(subset=('sp_id'))
    df_meta = pd.merge(df_tracks, df_sp_feat, on='sp_id')

    feats = ['tempo', 'danceability', 'acousticness', 'instrumentalness']
    header = ['count', 'mean', 'std', 'min', 'q1', 'q2', 'q3', 'IQR', 'max']

    for feat in feats:
        print("\n###########  {}".format(feat))

        diffs_joint = []
        tracks_feat_plot = []
        for lists in [list_div, list_not_div]:
            tracks_feat = df_meta.loc[df_meta.yt_id.isin(
                [item for elem in lists for item in elem])][feat]
            tracks_feat_plot.append(tracks_feat)
            count, mean, std, _min, q1, q2, q3, _max = tracks_feat.describe()
            print(tabulate([[count, mean, std, _min, q1, q2, q3, q3-q1, _max]],
                  headers=header))

            medians = []
            for tracklist in lists:
                median_list = df_meta.loc[df_meta.yt_id.isin(tracklist)][feat]
                medians.append(median_list.median())

            diffs = []
            for i1, i2 in combinations(medians, 2):
                diffs.append(abs(i1-i2))

            diffs_joint.append(diffs)

        df = pd.concat(tracks_feat_plot, axis=1)
        df.columns = ['High Diversity', 'Low Diversity']
        ax = df.boxplot()
        ax.get_figure().suptitle(t=feat, fontsize=16);
        plt.show()
        test_significance(diffs_joint[0], diffs_joint[1])


if __name__ == "__main__":

    df_tracks = pd.read_csv(TRACKS, delimiter='\t')

    (list_div, list_not_div,
        list_div_genres, list_not_div_genres) = import_lists()

    embeddings, filenames = import_embeddings(EMB_DIR, 'effnet_tsne', 2)
    embeddings = np.vstack(embeddings)
    DistMatrix = cdist(embeddings, embeddings, 'euclidean')

    list_div_idxs, list_not_div_idxs = find_indexes(list_div,
                                                    list_not_div,
                                                    filenames)

    plot_lists(embeddings, list_div_idxs, list_not_div_idxs,
               list_div_genres, list_not_div_genres)

    # test_distances()

    # test_features()
