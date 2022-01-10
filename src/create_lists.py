#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import numpy as np
import pandas as pd

from datetime import datetime
from itertools import combinations

from scipy.spatial.distance import cdist

now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")

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

LIST_DIV = "../data/lists/track_list_div_{}.csv".format(date_time)
LIST_NOT_DIV = "../data/lists/track_list_not_div_{}.csv".format(date_time)


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
            nn_dists.append(DistMatrix[n1, n2])

        avg_dists.append((i, np.average(nn_dists)))

    sort_avg_dists = sorted(avg_dists, key=lambda x: x[1])

    return sort_avg_dists


def create_div_list(num_list, sort_avg_dists, df_tracks, filenames):
    """
    """
    maingenres = df_tracks.maingenre.unique().tolist()

    nns = []
    nns_found = []
    for mgenre in maingenres:
        for c, (track_idx, dist) in enumerate(sort_avg_dists):
            yt_id = filenames[track_idx]

            if df_tracks[df_tracks.yt_id == yt_id].maingenre.item() == mgenre:

                nn = DistMatrix[track_idx].argsort()[:4]
                df_list = df_tracks[df_tracks.yt_id.isin(
                    [filenames[x] for x in nn])]
                list_genres = df_list.maingenre

                if len(list_genres.unique()) > 3:
                    continue

                if not any(map(lambda v: v in nns_found, nn)):
                    nns.append(nn)
                    nns_found.extend(nn)
                    break

    return nns, maingenres


def create_not_div_lists(num_list, sort_avg_dists, df_tracks, filenames):
    """
    """
    genres_allowed = ['trance']

    nns_found = []
    maingenres = []
    nns = []

    for c, (track_idx, dist) in enumerate(sort_avg_dists):

        nn = DistMatrix[track_idx].argsort()[:4]
        df_list = df_tracks[df_tracks.yt_id.isin([filenames[x] for x in nn])]
        list_genres = df_list.maingenre
        most_comm_genre = list_genres.value_counts().index.tolist()[0]

        if most_comm_genre not in genres_allowed:
            continue
        elif len(list_genres.unique()) > 3:
            continue
        elif not any(map(lambda v: v in nns_found, nn)):
            nns.append(nn)
            nns_found.extend(nn)
            maingenres.append(most_comm_genre)

        if len(nns) == num_list:
            break

    return nns, maingenres


def write_lists(outfile, tracklists, tracklist_genres):
    """
    """
    with open(outfile, 'w+') as outf:
        _writer = csv.writer(outf)
        for i in range(num_list):
            row = [filenames[x] for x in tracklists[i]]
            row.append(tracklist_genres[i])
            _writer.writerow(row)

    print("Created: {}".format(outfile))


if __name__ == "__main__":

    num_list = 20

    df_tracks = pd.read_csv(TRACKS, delimiter='\t')

    embeddings, filenames = import_embeddings(EMB_DIR, 'musicnn', 200)
    embeddings = np.vstack(embeddings)

    DistMatrix = cdist(embeddings, embeddings, 'cosine')

    sort_avg_dists = sort_tracks_by_distance(DistMatrix)

    nns_div, nns_div_genres = create_div_list(num_list,
                                              sort_avg_dists,
                                              df_tracks,
                                              filenames)

    nns, nns_genres = create_not_div_lists(num_list,
                                           sort_avg_dists,
                                           df_tracks,
                                           filenames)

    if len(nns_div) < num_list or len(nns) < num_list:
        raise Exception("{} - {}".format(len(nns_div), len(nns)))

    write_lists(LIST_DIV, nns_div, nns_div_genres)
    write_lists(LIST_NOT_DIV, nns, nns_genres)
