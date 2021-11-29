#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import random
import json 
import pandas as pd

from operator import itemgetter
from itertools import combinations

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, euclidean, cdist

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import get_cmap
from matplotlib.ticker import PercentFormatter


EMB_DIR = "/home/lorenzo/Data/longterm_data/effnet_filtered"
ESS_DIR = "/home/lorenzo/Data/longterm_data/test_audio/essentia"
METADATA = "/home/lorenzo/Workspace/longterm-musdiv/data/track_list_yt_20210923.csv"
MAP_GENRE = "/home/lorenzo/Workspace/longterm-musdiv/data/map_genres.csv"

def import_embeddings(emb_dir):
    """
    """
    # Import embeddings
    embeddings = []
    fnames = []
    for file in sorted(os.listdir(emb_dir)):
        fname, exts = os.path.splitext(file)
        if exts == '.npy':
            embedding = np.load(os.path.join(emb_dir, file))
            if len(embedding) != 200: # Sanity Check
                print(fname)
            else:
                embeddings.append(embedding)
                fnames.append(fname)

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

    return embeddings_reduced

def get_centroid(embeddings_reduced, fnames):
    """
    """
    # Get centroid
    emb_x = list(map(itemgetter(0), embeddings_reduced))
    emb_y = list(map(itemgetter(1), embeddings_reduced))
    C_x = np.mean(emb_x)
    C_y = np.mean(emb_y)

    # Get Track max dist from Centroid
    dists = [euclidean(x, [C_x, C_y]) for x in embeddings_reduced]
    max_dist = np.max(dists)
    imax_dist = dists.index(max_dist)
    # fnames[imax_dist] = fnames[imax_dist]+ '- MAX'
    print("Max dist = {}".format(max_dist))

    return emb_x, emb_y, C_x, C_y, max_dist

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

def import_features(feat_dir):
    """
    """
    # Store Dict with features 
    DictFeat = {}
    fnames = []
    for file in sorted(os.listdir(feat_dir)):
        fname, exts = os.path.splitext(file)
        if exts == '.json':
            with open(os.path.join(feat_dir, file), 'r') as inf:
                data = json.load(inf)

            DictFeat[fname] = {}
            DictFeat[fname]['bpm'] = data['rhythm']['bpm']
            DictFeat[fname]['dance'] = data['rhythm']['danceability']
            DictFeat[fname]['timbre'] = data['highlevel']['timbre']['all']['dark']
            DictFeat[fname]['instr'] = data['highlevel']['voice_instrumental']['all']['instrumental']
            DictFeat[fname]['gen_voice'] = data['highlevel']['gender']['all']['female']
            fnames.append(fname)


    print("Number of tracks feature found: {}".format(len(DictFeat))) 
    
    return DictFeat, fnames

def plot_features(DictFeat, fnames):
    """
    """
    fig, ax = plt.subplots(3,2)
    feat = [DictFeat[x]['bpm'] for x in DictFeat]
    ax[0, 0].hist(feat, alpha=0.7, rwidth=0.85, weights=np.ones(len(feat)) / len(feat))
    ax[0, 0].set_ylabel('Track Frequency')
    # ax[0, 0].set_xlabel('BPM')
    ax[0, 0].set_title('BPM Track distribution')
    ax[0, 0].text(150, 25, r'$\mu={:.2f}, \sigma={:.2f}$'.format(np.mean(feat), np.std(feat)))
    ax[0, 0].grid(axis='y', alpha=0.75)
    ax[0, 0].yaxis.set_major_formatter(PercentFormatter(1))
    # plt.show()

    # fig, ax = plt.subplots()
    feat = [DictFeat[x]['dance'] for x in DictFeat]
    ax[0, 1].hist(feat, alpha=0.7, rwidth=0.85, color='#baba07', weights=np.ones(len(feat)) / len(feat))
    ax[0, 1].set_ylabel('Track Frequency')
    # ax[0, 1].set_xlabel('Danceability')
    ax[0, 1].set_title('Danceability Track distribution')
    ax[0, 1].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(np.mean(feat), np.std(feat)))
    ax[0, 1].grid(axis='y', alpha=0.75)
    ax[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
    # plt.show()

    # fig, ax = plt.subplots()
    feat = [DictFeat[x]['timbre'] for x in DictFeat]
    ax[1, 0].hist(feat, alpha=0.7, rwidth=0.85, color='#216045', weights=np.ones(len(feat)) / len(feat))
    ax[1, 0].set_ylabel('Track Frequency')
    # ax[1, 0].set_xlabel('P(Darkness)')
    ax[1, 0].set_title('P(Darkness) Track distribution')
    ax[1, 0].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(np.mean(feat), np.std(feat)))
    ax[1, 0].grid(axis='y', alpha=0.75)
    # ax[1, 0].xticks(np.arange(0,1, 0.1)+0.05)
    ax[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    # plt.show()

    # fig, ax = plt.subplots()
    feat = [DictFeat[x]['instr'] for x in DictFeat]
    ax[1, 1].hist(feat, alpha=0.7, rwidth=0.85, color='#fb0a66', weights=np.ones(len(feat)) / len(feat))
    ax[1, 1].set_ylabel('Track Frequency')
    # ax[1, 1].set_xlabel('P(Instrumentalness)')
    ax[1, 1].set_title('P(Instrumentalness) Track distribution')
    ax[1, 1].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(np.mean(feat), np.std(feat)))
    ax[1, 1].grid(axis='y', alpha=0.75)
    # plt.xticks(np.arange(0,1, 0.1)+0.05)
    ax[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    # plt.show()

    # fig, ax = plt.subplots()
    feat = [DictFeat[x]['gen_voice'] for x in DictFeat]
    ax[2, 0].hist(feat, alpha=0.7, rwidth=0.85, color='#0504aa', weights=np.ones(len(feat)) / len(feat))
    ax[2, 0].set_ylabel('Track Frequency')
    # ax[2, 0].set_xlabel('P(Female voice)')
    ax[2, 0].set_title('P(Female Voice) Track distribution')
    ax[2, 0].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(np.mean(feat), np.std(feat)))
    # ax[1, 1].xticks(np.arange(0,1, 0.1)+0.05)
    ax[2, 0].grid(axis='y', alpha=0.75)
    ax[2, 0].yaxis.set_major_formatter(PercentFormatter(1))
    # plt.show()

    # fig, ax = plt.subplots()
    ax[2, 1].scatter([DictFeat[x]['instr'] for x in fnames], [DictFeat[x]['gen_voice'] for x in fnames], c = [DictFeat[x]['gen_voice'] for x in fnames], cmap="RdYlBu_r")
    ax[2, 1].set_ylabel('P(Female)')
    ax[2, 1].set_xlabel('P(Instrumentalness)')
    # ax[2, 1].set_title('Gender Voice X Instrumentalness')
    plt.show()

def import_metadata(meta_file):
    """
    """
    df = pd.read_csv(meta_file, delimiter='\t')
    df_map_genre = pd.read_csv(MAP_GENRE)

    df['yt_id'] = [x.split('?v=')[1] for x in df['yt_link']]
    df_new = pd.merge(df, df_map_genre, on='genre')

    genres = df_new['maingenre'].unique()

    return df_new, genres

def plot_embeddings_genre(df, genres, fnames):
    """
    """
    cmap = plt.cm.get_cmap('tab20', len(genres))

    X = []
    Y = []
    Z = []

    for i, label in enumerate(fnames):
        Z.append(df[df['yt_id'] == label]['maingenre'].values[0])
        X.append(emb_x[i])
        Y.append(emb_y[i])

    C = [cmap(np.where(genres == z))[0][0] for z in Z]
    fig, ax = plt.subplots(figsize=(7,7))
    for x, y, c, l in zip(X, Y, C, Z):
        ax.scatter(x, y, color=c, vmin=-2, label=l)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


if __name__ == "__main__":

    embeddings, fnames = import_embeddings(EMB_DIR)
    embeddings_reduced = reduce_embeddings(embeddings)
    emb_x, emb_y, C_x, C_y, max_dist = get_centroid(embeddings_reduced, fnames)

    # # Compute pairwise distances
    # DistMatrix = cdist(embeddings_reduced, embeddings_reduced, metric='euclidean')

    # sort_avg_dists = sort_tracks_by_distance(DistMatrix)

    # nns = create_lists(sort_avg_dists)

    df_meta, genres = import_metadata(METADATA)

    plot_embeddings_genre(df_meta, genres, fnames)




    # # Plot
    # fig, ax = plt.subplots()
    # ax.scatter(emb_x, emb_y)
    # for i, label in enumerate(fnames):
    #     plt.annotate(label, (emb_x[i], emb_y[i]))
    # ax.scatter(C_x, C_y)
    # ax.scatter([emb_x[i] for i in nns[0]], [emb_y[i] for i in nns[0]])
    # ax.scatter([emb_x[i] for i in nns[1]], [emb_y[i] for i in nns[1]])
    # ax.scatter([emb_x[i] for i in nns[2]], [emb_y[i] for i in nns[2]])
    # ax.scatter([emb_x[i] for i in nns[3]], [emb_y[i] for i in nns[3]])

    # plt.annotate('Centroid', (C_x, C_y))
    # cir = plt.Circle((C_x, C_y), max_dist, color='r',fill=False)
    # ax.set_aspect('equal', adjustable='datalim')
    # ax.add_patch(cir)
    # plt.show()


    # # Get features from essentia and plot
    # DictFeat, fnames = import_features(ESS_DIR)
    # plot_features(DictFeat, fnames)











    # Plot Distance Matrix
    # plt.figure()
    # ax = plt.subplot()
    # im = ax.imshow(DistMatrix, alpha=0.8, cmap='inferno')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    # plt.show()