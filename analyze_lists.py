#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import csv
import numpy as np

from itertools import combinations
from tabulate import tabulate
from scipy.spatial.distance import cdist, euclidean, cosine
from operator import itemgetter
from scipy import stats
from mannwhitney import mannWhitney


METADATA_ENRICH = "data/filtered_tracks_enriched_20211124.csv"

creation_time = "20211229_173136"
LIST_DIV = "data/lists/track_list_div_{}.csv".format(creation_time)
LIST_NOT_DIV = "data/lists/track_list_not_div_{}.csv".format(creation_time)


def import_lists():
    """
    """

    list_div, list_not_div = [], []

    with open(LIST_DIV, 'r') as inf1, open(LIST_NOT_DIV, 'r') as inf2:
        _reader1 = csv.reader(inf1)
        _reader2 = csv.reader(inf2)

        for row in _reader1:
            list_div.append(row)
        for row in _reader2:
            list_not_div.append(row)

    return list_div, list_not_div


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
    print("Medians -> ", np.median(list1), np.median(list2))
    print("Mann Whitney -> Significance: {}; U-statistics: {}, EffectSize: {}".format(
                                        MU.significance, MU.u, MU.effectsize))

    tStat, pValue = stats.ttest_ind(list1, list2)
    print("T-Test -> P-Value:{0} T-Statistic:{1}".format(pValue,tStat))


def test_distances():
    """
    """
    t_embs = [full, pca, tsne]
    l_embs = ["FULL", "PCA", "TSNE"]

    print("Distance from geometric median")
    for t_emb, l_emb in zip(t_embs, l_embs):
        print("###########  {}".format(l_emb))

        div_gmedian = geometric_median(np.vstack(itemgetter(*div_idxs)(t_emb)))
        div_dists = []
        for idx in div_idxs:
            dist = cosine(div_gmedian, t_emb[idx])
            div_dists.append(dist)

        not_div_gmedian = geometric_median(np.vstack(itemgetter(*not_div_idxs)(t_emb)))
        not_div_dists = []
        for idx in not_div_idxs:
            dist = cosine(not_div_gmedian, t_emb[idx])
            not_div_dists.append(dist)

        test_significance(div_dists, not_div_dists)


    print("\nInter list average distance")
    for t_emb, l_emb in zip(t_embs, l_embs):
        print("###########  {}".format(l_emb))
        avg_inter_list_1 = []
        for c1, l1 in enumerate(list_div_idxs):
            for c2, l2 in enumerate(list_div_idxs):
                if c1 <= c2:
                    continue
                emb1 = itemgetter(*l1)(t_emb)
                emb2 = itemgetter(*l2)(t_emb)
                distances = cdist(emb1, emb2, 'cosine')
                avg_dist = np.average(distances[np.nonzero(np.triu(distances))])
                avg_inter_list_1.append(avg_dist)

        avg_inter_list_2 = []
        for c1, l1 in enumerate(list_not_div_idxs):
            for c2, l2 in enumerate(list_not_div_idxs):
                if c1 <= c2:
                    continue
                emb1 = itemgetter(*l1)(t_emb)
                emb2 = itemgetter(*l2)(t_emb)
                distances = cdist(emb1, emb2, 'cosine')
                avg_dist = np.average(distances[np.nonzero(np.triu(distances))])
                avg_inter_list_2.append(avg_dist)

        test_significance(avg_inter_list_1, avg_inter_list_2)



    print("\nIntra list average distance")
    for t_emb, l_emb in zip(t_embs, l_embs):
        print("###########  {}".format(l_emb))
        avg_intra_list_1 = []
        for idxs in list_div_idxs:
            emb = itemgetter(*idxs)(t_emb)
            distances = cdist(emb, emb, 'cosine')
            avg_dist = np.average(distances[np.nonzero(np.triu(distances))])
            avg_intra_list_1.append(avg_dist)
        

        avg_intra_list_2 = []
        for idxs in list_not_div_idxs:
            emb = itemgetter(*idxs)(t_emb)
            distances = cdist(emb, emb, 'cosine')
            avg_dist = np.average(distances[np.nonzero(np.triu(distances))])
            avg_intra_list_2.append(avg_dist)
        
        test_significance(avg_intra_list_1, avg_intra_list_2)    



if __name__ == "__main__":

    df_meta = pd.read_csv(METADATA_ENRICH)
    full = np.load('data/emb/full.npy')
    pca = np.load('data/emb/pca.npy')
    tsne = np.load('data/emb/tsne.npy')
    fnames = np.load('data/emb/fnames.npy')

    list_div, list_not_div = import_lists()

    # Get nested list of indexes
    list_div_idxs = []
    for l in list_div:
        list_div_idxs.append([np.where(fnames == track)[0].item() for track in l])

    list_not_div_idxs = []
    for l in list_not_div:
        list_not_div_idxs.append([np.where(fnames == track)[0].item() for track in l])

    # Get flat list index
    div_idxs = [item for elem in list_div_idxs for item in elem]
    not_div_idxs = [item for elem in list_not_div_idxs for item in elem]


    test_distances()


    feats = ['bpm', 'dance', 'timbre', 'instr', 'voice']
    header = ["count", "mean", "std", "min", "q1", "q2", "q3", "IQR", "max"]

    for feat in feats:
        print("\n###########  {}".format(feat))
        
        median = df_meta.loc[df_meta.yt_id.isin([item for elem in list_div for item in elem])][feat].median()
        count, mean, std, _min, q1, q2, q3, _max = df_meta.loc[df_meta.yt_id.isin([item for elem in list_div for item in elem])][feat].describe()
        print(tabulate([[count, mean, std, _min, q1, q2, q3, q3-q1, _max]], headers=header))

        medians_1 = []
        for tracklist in list_div:
            median_list = df_meta.loc[df_meta.yt_id.isin(tracklist)][feat].median()
            medians_1.append(median_list)

        diffs1 = []
        for i1, i2 in combinations(medians_1, 2):
            diffs1.append(abs(i1-i2))

        median = df_meta.loc[df_meta.yt_id.isin([item for elem in list_not_div for item in elem])][feat].median()
        count, mean, std, _min, q1, q2, q3, _max = df_meta.loc[df_meta.yt_id.isin([item for elem in list_not_div for item in elem])][feat].describe()
        print(tabulate([[count, mean, std, _min, q1, q2, q3, q3-q1, _max]], headers=header))

        medians_2 = []
        for tracklist in list_not_div:
            median_list = df_meta.loc[df_meta.yt_id.isin(tracklist)][feat].median()
            medians_2.append(median_list)
        diffs2   = []
        for i1, i2 in combinations(medians_2, 2):
            diffs2.append(abs(i1-i2))


        test_significance(diffs1, diffs2)    

