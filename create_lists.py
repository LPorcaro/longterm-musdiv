#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import csv
from datetime import datetime

from operator import itemgetter
from itertools import combinations

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean, cdist

now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")

marker_types = [".", "o", "v", "^", "<", 
                ">", "1", "2", "3", "4",
                "8", "s", "p", "P", "h",
                "H", "+", "x", "X", "D",
                ".", "o", "v", "^", "<", '1']


ESSENTIA_DIR = "/home/lorenzo/Data/longterm_data/features/"
EMB_DIR = "../data/embeddings/{}"
TRACKS = "../data/input/random_tracklist_20220104.csv"
TRACKS_FEAT = "../data/input/tracklist_features_20220104.csv"

LIST_DIV = "data/lists/track_list_div_{}.csv".format(date_time)
LIST_NOT_DIV = "data/lists/track_list_not_div_{}.csv".format(date_time)



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
    genres_allowed = ['techno', 'trance', 'hardcore', 'hardstyle', 'house']


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
            elif not all(df_meta.iloc[nn].bpm > 110):
                continue
            elif not all(df_meta.iloc[nn].timbre > 0.45):
                continue
            elif not (all(df_meta.iloc[nn].dance > 0.9)):
                continue
            elif not all(df_meta.iloc[nn].instr < 0.85):
                continue
            elif not all(df_meta.iloc[nn].voice > 0.1):
                continue
            else:
                genres_found.append(most_comm_genre)

        # Check if tracks already in other lists
        if not any(map(lambda v: v in tracks_found, nn)):
            nns.append(nn)
            tracks_found.extend(nn)

        if len(nns) == num_list:
            break


    # # Write track lists
    # if diverse:
    #     outfile = LIST_DIV
    # else:
    #     outfile = LIST_NOT_DIV

    # with open(outfile, 'w+') as outf:
    #     _writer = csv.writer(outf)
    #     for i in range(num_list):
    #         _writer.writerow([df.iloc[c].yt_id for c in nns[i]])

    # print("Created: {}".format(outfile))

    return nns, genres_found


if __name__ == "__main__":

    num_list = 20

    embeddings, filenames = import_embeddings(EMB_DIR, 'musicnn_tsne', 2)

    embeddings = np.vstack(embeddings)
    emb_x = list(map(itemgetter(0), embeddings))
    emb_y = list(map(itemgetter(1), embeddings))

    C_x, C_y, max_dist, imax_dist = get_centroid(embeddings)

    DistMatrix = cdist(embeddings, embeddings, 'cosine')

    sort_avg_dists = sort_tracks_by_distance(DistMatrix)

    print(sort_avg_dists[:30])

    # nns_div, genres_found_div = create_lists(num_list, sort_avg_dists, df_meta, diverse=True)
    # nns, genres_found = create_lists(num_list, sort_avg_dists, df_meta, diverse=False)

    # if len(nns_div) < num_list or len(nns) < num_list:
    #       raise Exception("{} - {}".format(len(nns_div), len(nns)))

    # # Plot
    # fig, (ax1, ax2) = plt.subplots(ncols=2)
    # ax1.scatter(C_x, C_y)
    # ax1.set_title('List non diversified')
    # for c in range(num_list):
    #     ax1.scatter([emb_x[i] for i in nns[c]], [emb_y[i] for i in nns[c]], marker='x')
    #     ax1.annotate(df_meta.iloc[nns[c][0]].maingenre, (emb_x[nns[c][0]], emb_y[nns[c][0]]))

    # ax2.scatter(C_x, C_y)
    # ax2.set_title('List diversified')
    # for c in range(num_list):
    #     ax2.scatter([emb_x[i] for i in nns_div[c]], [emb_y[i] for i in nns_div[c]], marker='x')
    #     ax2.annotate(df_meta.iloc[nns_div[c][0]].maingenre, (emb_x[nns_div[c][0]], emb_y[nns_div[c][0]]))


    # ax1.annotate('Centroid', (C_x, C_y))
    # cir = plt.Circle((C_x, C_y), max_dist, color='r',fill=False)
    # ax1.set_aspect('equal', adjustable='datalim')
    # ax1.add_patch(cir)
    # cir = plt.Circle((C_x, C_y), max_dist, color='r',fill=False)
    # ax2.set_aspect('equal', adjustable='datalim')
    # ax2.add_patch(cir)
    # plt.show()