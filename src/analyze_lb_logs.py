#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import kendalltau
from datetime import datetime, timedelta
from collections import Counter, OrderedDict

WIKI_GENRES = "../data/input/wikipedia_EM_genres.csv"
FEAT_DIR = "../data/listenbrainz/feat"
INFO_DIR = "../data/listenbrainz/info"

end = datetime.strptime("20220216", "%Y%m%d")

NOT_EM_GENRE = ['permanent wave']

PRE = [datetime.strptime(x, "%Y%m%d") for x in ["20220216", "20220314"]]
COND = [datetime.strptime(x, "%Y%m%d") for x in ["20220315", "20220412"]]

TRACKS = "../data/input/random_tracklist_20220104.csv"
LIST_DIV = "../data/lists/track_list_div_20220208_112025.csv"
LIST_NOT_DIV = "../data/lists/track_list_not_div_20220208_112025.csv"



HEADER = ["username", "phase",
          "genres_unique", "genres_count", "EM_genres_unique", "EM_genres_unique(%)", 
          "EM_genres_count","EM_genres_count(%)", "EM_genres_top_head(%)", "gini_genres", "gini_EM_genres",
          "artists_unique", "artists_count", "EM_artists_unique", "EM_artists_unique(%)", 
          "EM_artists_count","EM_artists_count(%)", "EM_artists_top_head(%)", "gini_artists", "gini_EM_artists",
          "tracks_unique", "tracks_count", "EM_tracks_unique", "EM_tracks_unique(%)", 
          "EM_tracks_count","EM_tracks_count(%)", "EM_tracks_top_head(%)", "gini_tracks", "gini_EM_tracks",]


def gini(x):
    """
    """
    # The rest of the code requires numpy arrays.
    try:
        x = np.asarray(x)
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    except:
        return (-9)

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def search_EM_genres(df):
    """
    """
    EM_wiki_genres = []
    with open(WIKI_GENRES) as inf:
        _reader = csv.reader(inf)
        for row in _reader:
            EM_wiki_genres.append(row[0].lower())
    EM_wiki_genres.append("electronic")

    genres_all = []
    genres_EM = []
    for genre in df.genres.values:
        if pd.isnull(genre):
            continue
        elif genre == 'Empty':
            continue
        else:
            genres = genre.split(',')
            genres_all.extend(genres)
            g_EM_found = [x for x in genres if x in EM_wiki_genres]
            genres_EM.extend(g_EM_found)
            # String Match
            for g in genres:
                if g in g_EM_found:
                    continue
                for x in EM_wiki_genres:
                    if x in g:
                        genres_EM.append(g)

    genres_EM = [x for x in genres_EM if x not in NOT_EM_GENRE]

    return genres_EM, genres_all

def search_EM_artists(df, genres_EM):
    """
    """
    artist_all = []
    artist_EM = []
    for el in df[['artist_name', 'genres']].values:
        artist_all.extend(el[0].split(","))
        if pd.isnull(el[1]):
            continue
        elif [x for x in el[1].split(",") if x in genres_EM]:
            artist_EM.extend(el[0].split(","))

    return artist_EM, artist_all

def search_EM_tracks(df, genres_EM):
    """
    """
    tracks_all = []
    tracks_EM = []
    for el in df[['track_name', 'genres']].values:
        tracks_all.append(el[0])
        if pd.isnull(el[1]):
            continue
        elif [x for x in el[1].split(",") if x in genres_EM]:
            tracks_EM.append(el[0])

    return tracks_EM, tracks_all

def import_data(username):
    """
    """
    print("User '{}': Analyzing...".format(username))

    INFO_DIR_USER = os.path.join(INFO_DIR, username)
    df_info = pd.DataFrame()
    for info_file in os.listdir(INFO_DIR_USER):
        infile = os.path.join(INFO_DIR_USER, info_file)
        df_info_tmp = pd.read_csv(infile, parse_dates=[0], delimiter="\t")
        df_info = pd.concat([df_info, df_info_tmp], ignore_index=True)
    df_info = df_info.drop_duplicates()

    FEAT_DIR_USER = os.path.join(FEAT_DIR, username)
    df_feat = pd.DataFrame()
    for feat_file in os.listdir(FEAT_DIR_USER):
        infile = os.path.join(FEAT_DIR_USER, feat_file)
        df_feat_tmp = pd.read_csv(infile, delimiter="\t")
        df_feat = pd.concat([df_feat, df_feat_tmp], ignore_index=True)
    df_feat = df_feat.drop_duplicates(subset="sp_track_id")

    return df_feat, df_info

def process_data(df_info, df_feat):
    """
    """
    df_PRE = df_info[(df_info.listened_at >= PRE[0]) & (df_info.listened_at <= PRE[1])]
    df_COND = df_info[(df_info.listened_at >= COND[0]) & (df_info.listened_at <= COND[1])]
    df_PRE = pd.DataFrame({'sp_track_id': df_PRE.sp_track_id}).merge(df_feat, on="sp_track_id", how="left")
    df_COND = pd.DataFrame({'sp_track_id': df_COND.sp_track_id}).merge(df_feat, on="sp_track_id", how="left")

    return df_PRE, df_COND

def import_lists():
    """
    """
    df = pd.read_csv(TRACKS, delimiter='\t')

    a_div, t_div = [], []
    a_not_div, t_not_div = [], []

    with open(LIST_DIV, 'r') as inf1, open(LIST_NOT_DIV, 'r') as inf2:
        _reader1 = csv.reader(inf1)
        _reader2 = csv.reader(inf2)

        for row in _reader1:
            for el in row[:-1]:
                a_div.extend(df[df.yt_id==el].artist_name.values[0].split(","))
                t_div.append(df[df.yt_id==el].track_name.values[0])

        for row in _reader2:
            for el in row[:-1]:
                a_not_div.extend(df[df.yt_id==el].artist_name.values[0].split(","))
                t_not_div.append(df[df.yt_id==el].track_name.values[0])

    return a_div, t_div, a_not_div, t_not_div

def run(username, rows):
    """
    """
    df_feat, df_info = import_data(username)
    df_PRE, df_COND = process_data(df_info, df_feat)

    print("####### GENRES")
    for df, phase in zip([df_PRE, df_COND], ["PRE", "COND"]):
        genres_EM, genres_all = search_EM_genres(df)
        ca_EM = Counter(genres_EM)
        cg_all = Counter(genres_all)
        g_all_unq = int(len(cg_all))
        g_all_cnt = int(sum(cg_all.values()))
        g_EM_unq = int(len(ca_EM))
        g_EM_cnt = int(sum(ca_EM.values()))
        g_EM_unq_p = round(g_EM_unq*100/g_all_unq, 2)
        g_EM_cnt_p = round(g_EM_cnt*100/g_all_cnt, 2)
        top20p = round(len(cg_all)*20/100)
        g_top20_int = round(int(len(set(cg_all.most_common()[:top20p]).intersection(ca_EM.most_common())))*100/g_all_unq,2)
        g_gini_all = round(gini(pd.factorize(genres_all)[0]),2)
        g_gini_EM = round(gini(pd.factorize(genres_EM)[0]),2)
        print(ca_EM.most_common()[:10])
        print(cg_all.most_common()[:10])
        print("# Total Genres found: {} ({})".format(g_all_unq, g_all_cnt))
        print("# Total EM Genres found: {} ({})".format(g_EM_unq, g_EM_cnt))
        print("# Percentage EM Genres: {:.2f}% ({:.2f}%)".format(g_EM_unq_p, g_EM_cnt_p))
        print("# Percentage EM genres in the top 20%: {:.2f}%".format(g_top20_int))
        print("# Gini Genres: {} - Gini EM Genres: {}".format(g_gini_all, g_gini_EM))
        for topN in [5,10,20]:
            print("#Top {}".format(topN))
            jacc_sim = jaccard(ca_EM.most_common()[:topN], cg_all.most_common()[:topN])
            print("\tJaccard Similarity: {}".format(jacc_sim))

        print("####### ARTISTS")
        artists_EM, artists_all = search_EM_artists(df, genres_EM)
        ca_EM = Counter(artists_EM)
        ca_all = Counter(artists_all)
        a_all_unq = int(len(ca_all))
        a_all_cnt = int(sum(ca_all.values()))
        a_EM_unq = int(len(ca_EM))
        a_EM_cnt = int(sum(ca_EM.values()))
        a_EM_unq_p = round(a_EM_unq*100/a_all_unq,2)
        a_EM_cnt_p = round(a_EM_cnt*100/a_all_cnt,2)
        top20p = round(len(ca_all)*20/100)
        a_top20_int = round(int(len(set(ca_all.most_common()[:top20p]).intersection(ca_EM.most_common())))*100/a_all_unq,2)
        a_gini_all = round(gini(pd.factorize(artists_all)[0]),2)
        a_gini_EM = round(gini(pd.factorize(artists_EM)[0]),2)
        print(ca_EM.most_common()[:10])
        print(ca_all.most_common()[:10])
        print("# Total Artists found: {} ({})".format(a_all_unq, a_all_cnt))
        print("# Total EM Artists found: {} ({})".format(a_EM_unq, a_EM_cnt))
        print("# Percentage EM Artists: {:.2f}% ({:.2f}%)".format(a_EM_unq_p, a_EM_cnt_p))
        print("# Percentage EM Artists in the top 20%: {:.2f}%".format(a_top20_int))
        print("# Gini Artists: {} - Gini EM Artists: {}".format(a_gini_all, a_gini_EM))
        for topN in [5,10,20]:
            print("#Top {}".format(topN))
            jacc_sim = jaccard(ca_EM.most_common()[:topN], ca_all.most_common()[:topN])
            print("\tJaccard Similarity: {}".format(jacc_sim))

        print("####### TRACKS")
        tracks_EM, tracks_all = search_EM_tracks(df, genres_EM)
        ct_EM = Counter(tracks_EM)
        ct_all = Counter(tracks_all)
        t_all_unq = int(len(ct_all))
        t_all_cnt = int(sum(ct_all.values()))
        t_EM_unq = int(len(ct_EM))
        t_EM_cnt = int(sum(ct_EM.values()))
        top20p = round(len(ct_all)*20/100)
        t_EM_unq_p = round(t_EM_unq*100/t_all_unq, 2)
        t_EM_cnt_p = round(t_EM_cnt*100/t_all_cnt, 2)
        t_top20_int = round(int(len(set(ct_all.most_common()[:top20p]).intersection(ct_EM.most_common())))*100/t_all_unq,2)
        t_gini_all = round(gini(pd.factorize(tracks_all)[0]),2)
        t_gini_EM = round(gini(pd.factorize(tracks_EM)[0]),2)
        print(ct_EM.most_common()[:10])
        print(ct_all.most_common()[:10])
        print("# Total Tracks found: {} ({})".format(t_all_unq, t_all_cnt))
        print("# Total EM Tracks found: {} ({})".format(t_EM_unq, t_EM_cnt))
        print("# Percentage EM Tracks: {:.2f}% ({:.2f}%)".format(t_EM_unq_p, t_EM_cnt_p))
        print("# Percentage EM Tracks in the top 20%: {:.2f}%".format(t_top20_int))
        print("# Gini Tracks: {} - Gini EM Tracks: {}".format(t_gini_all, t_gini_EM))

        for topN in [5,10,20]:
            print("#Top {}".format(topN))
            jacc_sim = jaccard(ct_EM.most_common()[:topN], ct_all.most_common()[:topN])
            print("\tJaccard Similarity: {}".format(jacc_sim))

        row = [username, phase,
               g_all_unq, g_all_cnt, g_EM_unq, g_EM_unq_p, g_EM_cnt, g_EM_cnt_p, g_top20_int, g_gini_all, g_gini_EM,
               a_all_unq, a_all_cnt, a_EM_unq, a_EM_unq_p, a_EM_cnt, a_EM_cnt_p, a_top20_int, a_gini_all, a_gini_EM,
               t_all_unq, t_all_cnt, t_EM_unq, t_EM_unq_p, t_EM_cnt, t_EM_cnt_p, t_top20_int, t_gini_all, t_gini_EM]

        rows.append(row)

    return rows


def arg_parser():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", type=str, dest='username',
                        help="ListenBrainz Username")
    parser.add_argument("-i", "--input", type=str, dest='input_file',
                        help="Input file with ListenBrainz usernames")
    parser.add_argument("-o", "--output", type=str, dest='out_dir',
                        help="Output for JSON files")
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = arg_parser()

    username = args.username
    input_file = args.input_file
    out_dir = args.out_dir

    # rows = []
    # if username:
    #     rows = run(username, rows)
    #     # print([x - y for x,y in zip(rows[0][2:],rows[1][2:])])

    # elif input_file:
    #     infile = open(input_file, 'r')
    #     lines = infile.readlines()
    #     infile.close()

    #     for line in lines:
    #         username = line.strip()
    #         rows = run(username, rows)

    # with open("../data/listenbrainz/results/test2.csv", 'w+') as outf:
    #     _writer = csv.writer(outf)
    #     _writer.writerow(HEADER)
    #     for row in rows:
    #         _writer.writerow(row)



    df_feat, df_info = import_data('anaritaml')
    df_PRE, df_COND = process_data(df_info, df_feat)

    a_div, t_div, a_not_div, t_not_div = import_lists()

    genres_EM_1, genres_all_1 = search_EM_genres(df_PRE)
    genres_EM_2, genres_all_2 = search_EM_genres(df_COND)
    diff_EM_genres = int(len(set(genres_EM_2).difference(set(genres_EM_1))))
    try:
        diff_EM_genres_p = diff_EM_genres*100 / int(len(genres_EM_2))
    except ZeroDivisionError:
        diff_EM_genres_p = 0
    diff_all_genres = int(len(set(genres_all_2).difference(set(genres_all_1))))
    diff_all_genres_p = diff_all_genres*100 / len(set(genres_all_2))
    print(diff_EM_genres,diff_EM_genres_p, diff_all_genres, diff_all_genres_p)

    artists_EM_1, artists_all_1 = search_EM_artists(df_PRE, genres_EM_1)
    artists_EM_2, artists_all_2 = search_EM_artists(df_COND, genres_EM_2)
    diff_EM_artists = int(len(set(artists_EM_2).difference(set(artists_EM_1))))
    try:
        diff_EM_artists_p = diff_EM_artists*100 / int(len(artists_EM_2))
    except ZeroDivisionError:
        diff_EM_artists_p = 0        
    diff_all_artists = int(len(set(artists_all_2).difference(set(artists_all_1))))
    diff_all_artists_p = diff_all_artists*100 / len(set(artists_all_2))
    print(diff_EM_artists,diff_EM_artists_p, diff_all_artists, diff_all_artists_p)

    print(set(artists_EM_2).intersection(a_div))


    tracks_EM_1, tracks_all_1 = search_EM_tracks(df_PRE, genres_EM_1)
    tracks_EM_2, tracks_all_2 = search_EM_tracks(df_COND, genres_EM_2)
    diff_EM_tracks = int(len(set(tracks_EM_2).difference(set(tracks_EM_1))))
    try:
        diff_EM_tracks_p = diff_EM_tracks*100 / int(len(tracks_EM_2))
    except ZeroDivisionError:
        diff_EM_tracks_p = 0  
    diff_all_tracks = int(len(set(tracks_all_2).difference(set(tracks_all_1))))
    diff_all_tracks_p = diff_all_tracks*100 / len(set(tracks_all_2))
    print(diff_EM_tracks,diff_EM_tracks_p, diff_all_tracks, diff_all_tracks_p)

    print(set(tracks_EM_2).intersection(t_div))