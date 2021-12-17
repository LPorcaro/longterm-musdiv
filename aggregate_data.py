#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import json

EMB_DIR = "/home/lorenzo/Data/longterm_data/effnet_filtered_20211124/"
EMB_DIR_2 = "/home/lorenzo/Data/longterm_data/msd-musicnn-embeddings_20211124"

FEAT_DIR = "/home/lorenzo/Data/longterm_data/essentia_20211124/"
METADATA = "data/filtered_tracks_20211124.csv"
METADATA_ENRICH = "data/filtered_tracks_enriched_20211124.csv"
MAP_GENRE = "data/map_genres.csv"
GENRE_DIST_MATRIX = "data/genres_distances.npy"
GENRE_INDEX = "data/genres_index.csv"

COLUMNS = ['yt_id', 'genre', 'maingenre', 'artist_name', 'track_name',
           'track_pop', 'viewCount', 'track_isrc',
           'bpm', 'dance', 'timbre', 'instr', 'voice',
           'yt_link', 'pl_id', 'emb_path', 'emb_path_2', 'feat_path']


def import_metadata(meta_file):
    """
    """
    df = pd.read_csv(meta_file)
    df_map_genre = pd.read_csv(MAP_GENRE)
    df_genre_index = pd.read_csv(GENRE_INDEX)

    # Add YT id in separate column and remove duplicates
    df['yt_id'] = [x.split('?v=')[1] for x in df['yt_link']]
    df = df.drop_duplicates(subset=('yt_id'), keep='last')

    # Add Maingenre to genre
    df = pd.merge(df, df_map_genre, on='genre')
    df = df.sort_values(by=['maingenre', 'genre'])

    # Remove tracks with no embedding 1
    for yt_id in df.yt_id:
        file = yt_id + '.npy'
        file_path = os.path.join(EMB_DIR, file)
        if not os.path.exists(file_path):
            df = df.drop(df[df.yt_id == yt_id].index.values)

    # Remove tracks with no embedding 2
    for yt_id in df.yt_id:
        file = yt_id + '.npy'
        file_path = os.path.join(EMB_DIR_2, file)
        if not os.path.exists(file_path):
            df = df.drop(df[df.yt_id == yt_id].index.values)

    df['emb_path'] = [os.path.join(EMB_DIR, yt_id + '.npy') for yt_id in df.yt_id]
    df['emb_path_2'] = [os.path.join(EMB_DIR_2, yt_id + '.npy') for yt_id in df.yt_id]

    # Filter our specif maingenre
    df = df[df['maingenre'] != 'rock']

    return df, df_genre_index, df_map_genre


def import_features(feat_dir, df):
    """
    """
    DictFeat = {}
    for yt_id in df.yt_id:
        file = yt_id + '.json'
        file_path = os.path.join(feat_dir, file)
        # Check if feat file exists
        if not os.path.exists(file_path):
            df = df.drop(df[df.yt_id == yt_id].index.values)
        else:
            with open(file_path, 'r') as inf:
                data = json.load(inf)

            DictFeat[yt_id] = {}
            DictFeat[yt_id]['bpm'] = data['rhythm']['bpm']
            DictFeat[yt_id]['dance'] = data['rhythm']['danceability']
            DictFeat[yt_id]['timbre'] = data['highlevel']['timbre']['all']['dark']
            DictFeat[yt_id]['instr'] = data['highlevel']['voice_instrumental']['all']['instrumental']
            DictFeat[yt_id]['voice'] = data['highlevel']['gender']['all']['female']
            DictFeat[yt_id]['path'] = file_path

    # Add feat to dataframe
    df_meta['bpm'] = [DictFeat[yt_id]['bpm'] for yt_id in df.yt_id]
    df_meta['dance'] = [DictFeat[yt_id]['dance'] for yt_id in df.yt_id]
    df_meta['timbre'] = [DictFeat[yt_id]['timbre'] for yt_id in df.yt_id]
    df_meta['instr'] = [DictFeat[yt_id]['instr'] for yt_id in df.yt_id]
    df_meta['voice'] = [DictFeat[yt_id]['voice'] for yt_id in df.yt_id]
    df_meta['feat_path'] = [DictFeat[yt_id]['path'] for yt_id in df.yt_id]

    return df


if __name__ == "__main__":

    df_meta, df_genre_index, df_map_genre = import_metadata(METADATA)
    print("Found {} tracks with embeddings".format(len(df_meta)))

    df_meta = import_features(FEAT_DIR, df_meta)
    print("Found {} tracks with features".format(len(df_meta)))

    df_meta = df_meta[COLUMNS]

    df_meta.to_csv(METADATA_ENRICH, index=False)
