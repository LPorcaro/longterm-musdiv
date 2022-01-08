#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import pandas as pd
import seaborn as sns

from collections import OrderedDict, Counter
from operator import itemgetter

from scipy.stats.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, silhouette_samples

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import PercentFormatter

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


def import_features(feat_dir, df_tracks):
    """
    """
    df_sp_feat = pd.read_csv(TRACKS_FEAT, delimiter='\t')
    df_sp_feat = df_sp_feat.drop_duplicates(subset=('sp_id'))

    # Store Dict with features
    DictFeat = {}
    for yt_id in df_tracks.yt_id.values:
        filename = yt_id + '.json'
        file_path = os.path.join(feat_dir, filename)

        if os.path.exists(file_path):
            with open(file_path, 'r') as inf:
                data = json.load(inf)

            DictFeat[yt_id] = {}
            DictFeat[yt_id]['essentia'] = {}
            DictFeat[yt_id]['essentia']['bpm'] = (
                round(data['rhythm']['bpm'], 3))
            DictFeat[yt_id]['essentia']['dance'] = (
                round(data['rhythm']['danceability'], 3))
            DictFeat[yt_id]['essentia']['timbre'] = (
                round(data['highlevel']['timbre']['all']['dark'], 3))
            DictFeat[yt_id]['essentia']['instr'] = (
                round(data['highlevel']['voice_instrumental']
                      ['all']['instrumental'], 3))
            DictFeat[yt_id]['essentia']['gen_voice'] = (
                round(data['highlevel']['gender']['all']['female'], 3))
            DictFeat[yt_id]['essentia']['acoust'] = (
                round(data['highlevel']['mood_acoustic']
                      ['all']['acoustic'], 3))

            sp_id = df_tracks[df_tracks.yt_id == yt_id].sp_id.item()
            sp_id_feat = df_sp_feat[df_sp_feat.sp_id == sp_id]

            DictFeat[yt_id]['spotify'] = {}
            DictFeat[yt_id]['spotify']['bpm'] = (
                                sp_id_feat.tempo.item())
            DictFeat[yt_id]['spotify']['dance'] = (
                                sp_id_feat.danceability.item())
            DictFeat[yt_id]['spotify']['acoust'] = (
                                sp_id_feat.acousticness.item())
            DictFeat[yt_id]['spotify']['instr'] = (
                                sp_id_feat.instrumentalness.item())
            DictFeat[yt_id]['spotify']['speech'] = (
                                sp_id_feat.speechiness.item())
            DictFeat[yt_id]['spotify']['energy'] = (
                                sp_id_feat.energy.item())
            DictFeat[yt_id]['spotify']['valence'] = (
                                sp_id_feat.valence.item())

    print("Tracks feature found: {}".format(len(DictFeat)))

    return DictFeat


def feature_correlation(DictFeat):
    """
    """
    print("\nComputing feature correlation...")
    features = ['bpm', 'dance', 'instr', 'acoust']

    for feat in features:
        feat_sp = []
        feat_ess = []
        for yt_id in DictFeat.keys():
            feat_sp.append(DictFeat[yt_id]['spotify'][feat])
            feat_ess.append(DictFeat[yt_id]['essentia'][feat])

        rho, p = pearsonr(feat_sp, feat_ess)

        print("Feature {} --> rho:{:.3f}, p:{}".format(feat, rho, p))


def plot_essentia_features(DictFeat, fnames):
    """
    """
    fig, ax = plt.subplots(3, 2)
    feat = [DictFeat[x]['essentia']['bpm'] for x in DictFeat]
    ax[0, 0].hist(feat, alpha=0.7, rwidth=0.85,
                  weights=np.ones(len(feat)) / len(feat))
    ax[0, 0].set_ylabel('Frequency')
    ax[0, 0].set_title('BPM')
    ax[0, 0].text(150, 25, r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                                                np.mean(feat), np.std(feat)))
    ax[0, 0].grid(axis='y', alpha=0.75)
    ax[0, 0].yaxis.set_major_formatter(PercentFormatter(1))

    feat = [DictFeat[x]['essentia']['dance'] for x in DictFeat]
    ax[0, 1].hist(feat, alpha=0.7, rwidth=0.85, color='#baba07',
                  weights=np.ones(len(feat)) / len(feat))
    ax[0, 1].set_ylabel('Frequency')
    ax[0, 1].set_title('Danceability')
    ax[0, 1].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                                                np.mean(feat), np.std(feat)))
    ax[0, 1].grid(axis='y', alpha=0.75)
    ax[0, 1].yaxis.set_major_formatter(PercentFormatter(1))

    feat = [DictFeat[x]['essentia']['acoust'] for x in DictFeat]
    ax[1, 0].hist(feat, alpha=0.7, rwidth=0.85, color='#0504aa',
                  weights=np.ones(len(feat)) / len(feat))
    ax[1, 0].set_ylabel('Frequency')
    ax[1, 0].set_title('Acousticness')
    ax[1, 0].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                                                np.mean(feat), np.std(feat)))
    ax[1, 0].grid(axis='y', alpha=0.75)
    ax[1, 0].yaxis.set_major_formatter(PercentFormatter(1))

    feat = [DictFeat[x]['essentia']['instr'] for x in DictFeat]
    ax[1, 1].hist(feat, alpha=0.7, rwidth=0.85, color='#fb0a66',
                  weights=np.ones(len(feat)) / len(feat))
    ax[1, 1].set_ylabel('Frequency')
    ax[1, 1].set_title('Instrumentalness')
    ax[1, 1].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                                                np.mean(feat), np.std(feat)))
    ax[1, 1].grid(axis='y', alpha=0.75)
    ax[1, 1].yaxis.set_major_formatter(PercentFormatter(1))

    feat = [DictFeat[x]['essentia']['timbre'] for x in DictFeat]
    ax[2, 0].hist(feat, alpha=0.7, rwidth=0.85, color='#216045',
                  weights=np.ones(len(feat)) / len(feat))
    ax[2, 0].set_ylabel('Frequency')
    ax[2, 0].set_title('Darkness ')
    ax[2, 0].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                                                np.mean(feat), np.std(feat)))
    ax[2, 0].grid(axis='y', alpha=0.75)
    ax[2, 0].yaxis.set_major_formatter(PercentFormatter(1))

    v1 = [DictFeat[x]['essentia']['instr'] for x in fnames]
    v2 = [DictFeat[x]['essentia']['gen_voice'] for x in fnames]
    c = [DictFeat[x]['essentia']['gen_voice'] for x in fnames]
    ax[2, 1].scatter(v1, v2, c=c, cmap="RdYlBu_r")
    ax[2, 1].set_ylabel('Female Voice')
    ax[2, 1].set_xlabel('Instrumentalness')
    fig.suptitle("Essentia Feature Tracks Distribution")
    plt.show()


def plot_spotify_features(DictFeat, fnames):
    """
    """
    fig, ax = plt.subplots(3, 2)
    feat = [DictFeat[x]['spotify']['bpm'] for x in DictFeat]
    ax[0, 0].hist(feat, alpha=0.7, rwidth=0.85,
                  weights=np.ones(len(feat)) / len(feat))
    ax[0, 0].set_ylabel('Frequency')
    ax[0, 0].set_title('BPM')
    ax[0, 0].text(150, 25, r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                                                np.mean(feat), np.std(feat)))
    ax[0, 0].grid(axis='y', alpha=0.75)
    ax[0, 0].yaxis.set_major_formatter(PercentFormatter(1))

    feat = [DictFeat[x]['spotify']['dance'] for x in DictFeat]
    ax[0, 1].hist(feat, alpha=0.7, rwidth=0.85, color='#baba07',
                  weights=np.ones(len(feat)) / len(feat))
    ax[0, 1].set_ylabel('Frequency')
    ax[0, 1].set_title('Danceability')
    ax[0, 1].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                                                np.mean(feat), np.std(feat)))
    ax[0, 1].grid(axis='y', alpha=0.75)
    ax[0, 1].yaxis.set_major_formatter(PercentFormatter(1))

    feat = [DictFeat[x]['spotify']['acoust'] for x in DictFeat]
    ax[1, 0].hist(feat, alpha=0.7, rwidth=0.85, color='#216045',
                  weights=np.ones(len(feat)) / len(feat))
    ax[1, 0].set_ylabel('Frequency')
    ax[1, 0].set_title('Acousticness')
    ax[1, 0].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                                                np.mean(feat), np.std(feat)))
    ax[1, 0].grid(axis='y', alpha=0.75)
    ax[1, 0].yaxis.set_major_formatter(PercentFormatter(1))

    feat = [DictFeat[x]['spotify']['instr'] for x in DictFeat]
    ax[1, 1].hist(feat, alpha=0.7, rwidth=0.85, color='#fb0a66',
                  weights=np.ones(len(feat)) / len(feat))
    ax[1, 1].set_ylabel('Frequency')
    ax[1, 1].set_title('Instrumentalness')
    ax[1, 1].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                                                np.mean(feat), np.std(feat)))
    ax[1, 1].grid(axis='y', alpha=0.75)
    ax[1, 1].yaxis.set_major_formatter(PercentFormatter(1))

    feat = [DictFeat[x]['spotify']['energy'] for x in DictFeat]
    ax[2, 0].hist(feat, alpha=0.7, rwidth=0.85, color='#0504aa',
                  weights=np.ones(len(feat)) / len(feat))
    ax[2, 0].set_ylabel('Frequency')
    ax[2, 0].set_title('Energy')
    ax[2, 0].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                                                np.mean(feat), np.std(feat)))
    ax[2, 0].grid(axis='y', alpha=0.75)
    ax[2, 0].yaxis.set_major_formatter(PercentFormatter(1))

    feat = [DictFeat[x]['spotify']['valence'] for x in DictFeat]
    ax[2, 1].hist(feat, alpha=0.7, rwidth=0.85, color='#090909',
                  weights=np.ones(len(feat)) / len(feat))
    ax[2, 1].set_ylabel('Frequency')
    ax[2, 1].set_title('Valence')
    ax[2, 1].text(2, 27, r'$\mu={:.2f}, \sigma={:.2f}$'.format(
                                                np.mean(feat), np.std(feat)))
    ax[2, 1].grid(axis='y', alpha=0.75)
    ax[2, 1].yaxis.set_major_formatter(PercentFormatter(1))

    fig.suptitle("Spotify Feature Tracks Distribution")
    plt.show()


def silhouette_analysis(DistMatrix, df_tracks, filenames):
    """
    """
    genres = df_tracks['maingenre'].unique().tolist()

    Z = []
    for filename in filenames:
        Z.append(df_tracks[
            df_tracks['yt_id'] == filename]['maingenre'].values[0])

    # Silhouette analysis
    n_clusters = len(genres)
    silhouette_avg = silhouette_score(DistMatrix, Z)
    sample_silhouette_values = silhouette_samples(DistMatrix, Z)

    fig, ax = plt.subplots()
    y_lower = 5
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[
                                [j for j, x in enumerate(Z) if x == genres[i]]]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
            label=genres[i]+" ({:.2f})".format(
                np.average(ith_cluster_silhouette_values)),
        )

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("Silhouette plot for the various genres")
    ax.set_xlabel("Silhouette coefficient values")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])  # Clear the yaxis labels / ticks

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.1, 0.85))
    plt.show()


def plot_embeddings(DistMatrix, df_tracks, emb_x, emb_y, filenames):
    """
    """
    genres = df_tracks['maingenre'].unique().tolist()
    n_clusters = len(genres)

    Z = []
    for filename in filenames:
        Z.append(df_tracks[
            df_tracks['yt_id'] == filename]['maingenre'].values[0])

    # Plot embeddings
    fig, ax = plt.subplots()
    for x, y, l in zip(emb_x, emb_y, Z):
        g_idx = genres.index(l)
        color = cm.nipy_spectral(float(g_idx) / n_clusters)
        ax.scatter(x, y, color=color, vmin=-2, label=l,
                   marker=marker_types[g_idx])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = OrderedDict(sorted(by_label.items()))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.1, 1.05))
    fig.suptitle("Tracks Embedding Space")

    plt.show()


def plot_distance_matrix(DistMatrix, df_tracks, filenames):
    """
    """
    idxs = []
    for c, filename in enumerate(filenames):
        track_maingenre = df_tracks[
                    df_tracks.yt_id == filename].maingenre.item()
        idxs.append((c, track_maingenre))

    # Re-order matrix
    idxs.sort(key=lambda x: x[1])
    genres_idxs = list(zip(*idxs))[1]
    frequency = Counter(genres_idxs)
    od = OrderedDict(sorted(frequency.items()))
    idxs = list(zip(*idxs))[0]

    DistMatrixOrd = np.zeros(DistMatrix.shape)
    for c1, i1 in enumerate(idxs):
        for c2, i2 in enumerate(idxs):
            if c2 >= c1:
                DistMatrixOrd[c1, c2] = DistMatrixOrd[c2, c1] = (
                    DistMatrix[i1, i2])

    # Create ticks
    st = 0
    tk = []
    tk_m = []
    for genre in od.keys():
        tk.append(st+round(od[genre]/2)+1)
        tk_m.append(st+od[genre]-0.5)
        st += od[genre]

    DistMatrixN = DistMatrixOrd / np.max(DistMatrixOrd)

    # Plot Distance Matrix
    fig, ax = plt.subplots()
    im = ax.imshow(DistMatrixN, alpha=0.8, cmap='binary')
    ax.set_xticks(tk)
    ax.set_yticks(tk)
    ax.set_xticklabels(od.keys(), rotation=90)
    ax.set_yticklabels(od.keys(), rotation=0)
    ax.set_xticks(tk_m, minor=True)
    ax.set_yticks(tk_m, minor=True)
    ax.grid(which='minor', color='r', linestyle='--', linewidth=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    fig.suptitle("Embeddings Distance Matrix")

    plt.show()


def plot_blocks_matrix_feat(emb_x, emb_y, DictFeat, filenames):
    """
    """
    features = ['bpm', 'dance', 'instr', 'acoust']
    feat_plot = ['BPM', 'Danceability', 'Instrumentalness', 'Acousticness']

    # Analyze features in blocks
    x_blocks = [(r, r+10) for r in range(
                        int(np.floor(min(emb_x))), int(max(emb_x)), 10)]
    y_blocks = [(r, r+10) for r in range(
                        int(np.floor(min(emb_y))), int(max(emb_y)), 10)]

    # Assign tracks to blocks
    BDict = {}
    for e, (x, y, l) in enumerate(zip(emb_x, emb_y, filenames)):
        x_b = [c for c, xb in enumerate(
                            x_blocks) if x >= xb[0] and x <= xb[1]][0]
        y_b = [c for c, yb in enumerate(
                            y_blocks) if y >= yb[0] and y <= yb[1]][0]

        key = str(x_b) + '-' + str(y_b)

        if key not in BDict:
            BDict[key] = {}
            BDict[key]["labels"] = []
        BDict[key]["labels"].append(l)

    # Compute average features for blocks
    for key in BDict:
        for feat in features:
            BDict[key][feat] = np.average(
                [DictFeat[x]['spotify'][feat] for x in BDict[key]['labels']])

    # Plots
    fig, axs = plt.subplots(2, 2)
    axs = axs.reshape(-1)
    for c, ax in enumerate(axs):
        BMatrix = np.zeros((len(x_blocks), len(y_blocks)))
        for key in BDict:
            idx1, idx2 = [int(x) for x in key.split('-')]
            BMatrix[idx1, idx2] = BDict[key][features[c]]
        ax.set_title('{}'.format(feat_plot[c]))
        sns.heatmap(BMatrix, linewidths=.5, annot=True, cmap='summer',
                    fmt='.3g', xticklabels=False, yticklabels=False, ax=ax)

    fig.suptitle("Embeddings-Features Blocks Distribution")
    plt.show()


if __name__ == "__main__":

    df_tracks = pd.read_csv(TRACKS, delimiter='\t')

    DictFeat = import_features(ESSENTIA_DIR, df_tracks)
    feature_correlation(DictFeat)
    embeddings, filenames = import_embeddings(EMB_DIR, 'musicnn_tsne', 2)
    print("Embeddings found: {}".format(len(embeddings)))

    embeddings = np.vstack(embeddings)
    emb_x = list(map(itemgetter(0), embeddings))
    emb_y = list(map(itemgetter(1), embeddings))

    # Compute pairwise distances
    DistMatrix = cdist(embeddings, embeddings, 'cosine')

    # Silhouette analysis
    # silhouette_analysis(DistMatrix, df_tracks, filenames)

    # # # Plots
    plot_embeddings(DistMatrix, df_tracks, emb_x, emb_y, filenames)
    plot_distance_matrix(DistMatrix, df_tracks, filenames)
    plot_essentia_features(DictFeat, filenames)
    plot_spotify_features(DictFeat, filenames)
    plot_blocks_matrix_feat(emb_x, emb_y, DictFeat, filenames)
