#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import kendalltau, entropy
from datetime import datetime, timedelta
from collections import Counter, OrderedDict
from itertools import combinations
from wordcloud import WordCloud

import sys 
sys.path.insert(0, '../data/input/')
from lb_logs_genres import EM_MAP, NOT_EM_GENRE

WIKI_GENRES = "../data/input/wikipedia_EM_genres.csv"
FEAT_DIR = "../data/listenbrainz/feat"
INFO_DIR = "../data/listenbrainz/info"

end = datetime.strptime("20220216", "%Y%m%d")

## Pilot 
# PRE = [datetime.strptime(x, "%Y%m%d") for x in ["20220216", "20220314"]]
# COND = [datetime.strptime(x, "%Y%m%d") for x in ["20220315", "20220412"]]
# POST = [datetime.strptime(x, "%Y%m%d") for x in ["20220413", "20220510"]]

# PRE = [datetime.strptime(x, "%Y%m%d") for x in ["20220502", "20220529"]]
COND = [datetime.strptime(x, "%Y%m%d") for x in ["20220530", "20220626"]]
POST = [datetime.strptime(x, "%Y%m%d") for x in ["20220627", "20220725"]]
PRE = [datetime.strptime(x, "%Y%m%d") for x in ["20220502", "20220725"]]


HD = ["Manhento","mrndcn","HiddenTrack","andrespi","acarolinab","An_Oxygen_Consumer","MayaIaia4","benfica15951","aimonteiror","martagmp","dv248","RedPeppermint","5ea1c3f3e9fb240a9b243e18","olline","Bruno10000silva","axel22","Branky","Apex23","afonsoveiga","deniseterzi","Kyor","Dinossaralho","RicardoCoelho8","Akromancer","CarolFrancis","rqlmarim","Ramiroowww","Edu_Marques17","matiri95","bpin","Gon1507","mariiii","Gonçalo","jshadyc","itsmatild","Ana Wolfie","ddios","diogojf","60f1881264ff3a7f211d186a","filippo_malocco","ilvapleuvoir","Acemorais","dfilipa98","61141c855eb58d6a4e3d8d5b","Holt92","Clara319","annoyin9","fraaa","Bea_P","ChiaraLh","aritahierro","gabriverce","joaomdpaz","Alba_hontoria_96","TheMusicWeirdo"]
LD = ["vrnk98","sneakysoup","VeroBerardi","sarafdelgado","user112553","Ordep123","vampiregrave","tommyjarvis","janybatista","quicomedeiros","LP22","versilvia","ruiipereiraa","Luijinho","ruipedro3","Inês","mariana02","Lostveneto","berenike","pirolocams","JohnnyMichaels","anasofiamachado","Ile.96","giulispaggiari","ippo","SunshineT93","inxsc","Ruhdyy","Theossi2000","francisco98","ricxtreme","joaoms98","valxros","prolificuser","SofiiFreitas","saturnzbarzs","nowhereknows","BlueGiuly","popu","Chaps03","outofljfe","ritamont","mapu25","ossalatodipotassio","Vittoria","tiagofcneves","Solk","LBS","silvawestcoast","salvatore9","RickyLB","MLN","Francisca","brunix21","Capuchinho"]

TRACKS = "../data/input/random_tracklist_20220104.csv"
LIST_DIV = "../data/lists/track_list_div_20220208_112025.csv"
LIST_NOT_DIV = "../data/lists/track_list_not_div_20220208_112025.csv"

OUTFILE_1 = "../data/listenbrainz/results/logs_analysis_full.csv"
OUTFILE_2 = "../data/listenbrainz/results/logs_diff_analysis.csv"

HEADER = ["username", 'group',"phase",
          "genres_unique", "genres_count", "EM_genres_unique", "EM_genres_unique_p", 
          "EM_genres_count","EM_genres_count_p", "EM_genres_top_head_p", "Jaccard_top5_genres",
          "Jaccard_top10_genres", "Jaccard_top20_genres","gini_genres", "gini_EM_genres",
          "artists_unique", "artists_count", "EM_artists_unique", "EM_artists_unique_p", 
          "EM_artists_count","EM_artists_count_p", "EM_artists_top_head_p", "Jaccard_top5_artists",
          "Jaccard_top10_artists", "Jaccard_top20_artists", "gini_artists", "gini_EM_artists",
          "tracks_unique", "tracks_count", "EM_tracks_unique", "EM_tracks_unique_p", 
          "EM_tracks_count","EM_tracks_count_p", "EM_tracks_top_head_p", "Jaccard_top5_tracks",
          "Jaccard_top10_tracks", "Jaccard_top20_tracks","gini_tracks", "gini_EM_tracks"]

HEADER_DIFF = ['username', 'group', 'phase',
               'nov genres', 'nov genres_p', 'nov EM genres', 'nov EM genres_p',
               'nov artists', 'nov artists_p', 'nov EM artists', 'nov EM artists_p',
               'nov tracks', 'nov tracks_p', 'nov EM tracks', 'nov EM tracks_p',
               'match artists', 'match tracks']


def gini(x):
    """
    """
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    gini_index = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return gini_index

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
        if isinstance(el[0], float):
            continue
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
    df_POST = df_info[(df_info.listened_at >= POST[0]) & (df_info.listened_at <= POST[1])]
    df_PRE = pd.DataFrame({'sp_track_id': df_PRE.sp_track_id}).merge(df_feat, on="sp_track_id", how="left")
    df_COND = pd.DataFrame({'sp_track_id': df_COND.sp_track_id}).merge(df_feat, on="sp_track_id", how="left")
    df_POST = pd.DataFrame({'sp_track_id': df_POST.sp_track_id}).merge(df_feat, on="sp_track_id", how="left")

    return df_PRE, df_COND, df_POST

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
    print("User '{}': Analyzing logs".format(username))

    # Check file info
    if df_info.empty:
        print("### User File not found : {}".format(username))
        return rows

    # Check data for each phase not empty
    df_PRE, df_COND, df_POST = process_data(df_info, df_feat)
    if df_PRE.empty or df_COND.empty or df_POST.empty:
        print("### User data not complete : {}".format(username))
        return rows

    # Check user group
    if username not in HD + LD:
        print("### User not found : {}".format(username))
        return rows
    elif username in HD:
        group = 'HD'
    elif username in LD:
        group = 'LD'     
    
    # for df, phase in zip([df_PRE, df_COND, df_POST], ["PRE", "COND", "POST"]):
    for df, phase in zip([df_PRE], ["PRE"]):
        # print("##### {} #####".format(phase))
        
        genres_EM, genres_all = search_EM_genres(df)
        if genres_all == []:
            row = [username, group, phase] + [0 for x in range(36)]
        else:
            # print("GENRES")
            cg_EM = Counter(genres_EM)
            cg_all = Counter(genres_all)
            g_all_unq = int(len(cg_all))
            g_all_cnt = int(sum(cg_all.values()))
            g_EM_unq = int(len(cg_EM))
            g_EM_cnt = int(sum(cg_EM.values()))
            g_EM_unq_p = round(g_EM_unq*100/g_all_unq, 2)
            g_EM_cnt_p = round(g_EM_cnt*100/g_all_cnt, 2)
            top20p = round(len(cg_all)*20/100)
            g_top20_int = round(int(len(set(cg_all.most_common()[:top20p]).intersection(cg_EM.most_common())))*100/g_all_unq,2)
            g_gini_all = None
            if genres_all:
                index = gini([*cg_all.values()])
                if not np.isnan(index):
                    g_gini_all = round(index,2)
            g_gini_EM = None    
            if genres_EM:
                index = gini([*cg_EM.values()])
                if not np.isnan(index):
                    g_gini_EM = round(index,2)                
            # print(cg_EM.most_common()[:10])
            # print(cg_all.most_common()[:10])
            # print("# Total Genres found: {} ({})".format(g_all_unq, g_all_cnt))
            # print("# Total EM Genres found: {} ({})".format(g_EM_unq, g_EM_cnt))
            # print("# Percentage EM Genres: {:.2f}% ({:.2f}%)".format(g_EM_unq_p, g_EM_cnt_p))
            # print("# Percentage EM genres in the top 20%: {:.2f}%".format(g_top20_int))
            # print("# Gini Genres: {} - Gini EM Genres: {}".format(g_gini_all, g_gini_EM))
            g_jac_top_5 = round(jaccard(cg_EM.most_common()[:5], cg_all.most_common()[:5]),2)
            g_jac_top_10 = round(jaccard(cg_EM.most_common()[:10], cg_all.most_common()[:10]),2)
            g_jac_top_20 = round(jaccard(cg_EM.most_common()[:20], cg_all.most_common()[:20]),2)

            # print("ARTISTS")
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
            a_gini_all = None
            if artists_all:
                index = gini([*ca_all.values()])
                if not np.isnan(index):
                    a_gini_all = round(index,2)

            a_gini_EM = None
            if artists_EM:
                index = gini([*ca_EM.values()])
                if not np.isnan(index):
                    a_gini_EM = round(index,2)

            # print(ca_EM.most_common()[:10])
            # print(ca_all.most_common()[:10])
            # print("# Total Artists found: {} ({})".format(a_all_unq, a_all_cnt))
            # print("# Total EM Artists found: {} ({})".format(a_EM_unq, a_EM_cnt))
            # print("# Percentage EM Artists: {:.2f}% ({:.2f}%)".format(a_EM_unq_p, a_EM_cnt_p))
            # print("# Percentage EM Artists in the top 20%: {:.2f}%".format(a_top20_int))
            # print("# Gini Artists: {} - Gini EM Artists: {}".format(a_gini_all, a_gini_EM))

            a_jac_top_5 = round(jaccard(ca_EM.most_common()[:5], ca_all.most_common()[:5]),2)
            a_jac_top_10 = round(jaccard(ca_EM.most_common()[:10], ca_all.most_common()[:10]),2)
            a_jac_top_20 = round(jaccard(ca_EM.most_common()[:20], ca_all.most_common()[:20]),2)

            # print("TRACKS")
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
            t_gini_all = None
            if tracks_all:
                index = gini([*ct_all.values()])
                if not np.isnan(index):
                    t_gini_all = round(index,2)
    
            t_gini_EM = None
            if tracks_EM:
                index = gini([*ct_EM.values()])
                if not np.isnan(index):
                    t_gini_EM = round(index,2)

            # print(ct_EM.most_common()[:10])
            # print(ct_all.most_common()[:10])
            # print("# Total Tracks found: {} ({})".format(t_all_unq, t_all_cnt))
            # print("# Total EM Tracks found: {} ({})".format(t_EM_unq, t_EM_cnt))
            # print("# Percentage EM Tracks: {:.2f}% ({:.2f}%)".format(t_EM_unq_p, t_EM_cnt_p))
            # print("# Percentage EM Tracks in the top 20%: {:.2f}%".format(t_top20_int))
            # print("# Gini Tracks: {} - Gini EM Tracks: {}".format(t_gini_all, t_gini_EM))

            t_jac_top_5 = round(jaccard(ct_EM.most_common()[:5], ct_all.most_common()[:5]),2)
            t_jac_top_10 = round(jaccard(ct_EM.most_common()[:10], ct_all.most_common()[:10]),2)
            t_jac_top_20 = round(jaccard(ct_EM.most_common()[:20], ct_all.most_common()[:20]),2)


            row = [username, group, phase,
                   g_all_unq, g_all_cnt, g_EM_unq, g_EM_unq_p, g_EM_cnt, g_EM_cnt_p, g_top20_int, g_jac_top_5, g_jac_top_10, g_jac_top_20, g_gini_all, g_gini_EM,
                   a_all_unq, a_all_cnt, a_EM_unq, a_EM_unq_p, a_EM_cnt, a_EM_cnt_p, a_top20_int, a_jac_top_5, a_jac_top_10, a_jac_top_20, a_gini_all, a_gini_EM,
                   t_all_unq, t_all_cnt, t_EM_unq, t_EM_unq_p, t_EM_cnt, t_EM_cnt_p, t_top20_int, t_jac_top_5, t_jac_top_10, t_jac_top_20, t_gini_all, t_gini_EM]

        rows.append(row)

    return rows

def run_diff(username, rows):
    """
    """
    df_feat, df_info = import_data(username)
    print("User '{}': Analyzing diff".format(username))

    if df_info.empty:
        print("### User File not found : {}".format(username))
        return rows

    df_PRE, df_COND, df_POST = process_data(df_info, df_feat)

    if df_PRE.empty or df_COND.empty or df_POST.empty:
        print("### User data not complete : {}".format(username))
        return rows

    a_hd, t_hd, a_ld, t_ld = import_lists()

    if username not in HD + LD:
        print("### User not found : {}".format(username))
        return rows
    elif username in HD:
        artist_tomatch = a_hd
        track_tomatch = t_hd
        group = 'HD'
    elif username in LD:
        artist_tomatch = a_ld
        track_tomatch = t_ld
        group = 'LD'     


    COMBO = [[df_PRE, df_COND], [pd.concat([df_PRE, df_COND], ignore_index=True), df_POST]]
    COMBO_LABEL = ["COND", "POST"]

    for (df_1, df_2), combo_l in zip(COMBO, COMBO_LABEL):

        if df_1.empty or df_2.empty:
            print("### User data not complete : {}".format(username))
            return rows

        # Genres
        genres_EM_1, genres_all_1 = search_EM_genres(df_1)
        genres_EM_2, genres_all_2 = search_EM_genres(df_2)

        if genres_all_2 == []:
            row = [username, group, combo_l] + [0 for x in range(14)]
        else:
            diff_EM_genres = int(len(set(genres_EM_2).difference(set(genres_EM_1))))
            try:
                diff_EM_genres_p = round(diff_EM_genres*100 / int(len(set(genres_EM_2))),2)
            except ZeroDivisionError:
                diff_EM_genres_p = 0
            diff_all_genres = int(len(set(genres_all_2).difference(set(genres_all_1))))
            diff_all_genres_p = round(diff_all_genres*100 / len(set(genres_all_2)),2)
            # print(diff_EM_genres, diff_EM_genres_p, diff_all_genres, diff_all_genres_p)

            # Artists
            artists_EM_1, artists_all_1 = search_EM_artists(df_1, genres_EM_1)
            artists_EM_2, artists_all_2 = search_EM_artists(df_2, genres_EM_2)
            diff_EM_artists = int(len(set(artists_EM_2).difference(set(artists_EM_1))))
            try:
                diff_EM_artists_p = round(diff_EM_artists*100 / int(len(set(artists_EM_2))),2)
            except ZeroDivisionError:
                diff_EM_artists_p = 0        
            diff_all_artists = int(len(set(artists_all_2).difference(set(artists_all_1))))
            diff_all_artists_p = round(diff_all_artists*100 / len(set(artists_all_2)),2)
            # print(diff_EM_artists,diff_EM_artists_p, diff_all_artists, diff_all_artists_p)

            # Tracks
            tracks_EM_1, tracks_all_1 = search_EM_tracks(df_1, genres_EM_1)
            tracks_EM_2, tracks_all_2 = search_EM_tracks(df_2, genres_EM_2)
            diff_EM_tracks = int(len(set(tracks_EM_2).difference(set(tracks_EM_1))))
            try:
                diff_EM_tracks_p = round(diff_EM_tracks*100 / int(len(set(tracks_EM_2))),2)
            except ZeroDivisionError:
                diff_EM_tracks_p = 0  
            diff_all_tracks = int(len(set(tracks_all_2).difference(set(tracks_all_1))))
            diff_all_tracks_p = round(diff_all_tracks*100 / len(set(tracks_all_2)),2)
            # print(diff_EM_tracks,diff_EM_tracks_p, diff_all_tracks, diff_all_tracks_p)

            artist_match = len(set(artists_EM_2).intersection(artist_tomatch))
            track_match = len(set(tracks_EM_2).intersection(track_tomatch))

            row = [username, group, combo_l,
                   diff_all_genres, diff_all_genres_p, diff_EM_genres, diff_EM_genres_p,
                   diff_all_artists, diff_all_artists_p, diff_EM_artists,diff_EM_artists_p,
                   diff_all_tracks, diff_all_tracks_p, diff_EM_tracks,diff_EM_tracks_p,
                   artist_match, track_match]


        rows.append(row)
    return rows

def run_qual(username, C_HD, C_LD):
    """
    """
    df_feat, df_info = import_data(username)
    print("User '{}': Analyzing logs".format(username))

    # Check file info
    if df_info.empty:
        print("### User File not found : {}".format(username))
        return 

    # Check data for each phase not empty
    df_PRE, df_COND, df_POST = process_data(df_info, df_feat)
    if df_PRE.empty or df_COND.empty or df_POST.empty:
        print("### User data not complete : {}".format(username))
        return 

    # Check user group
    if username not in HD + LD:
        print("### User not found : {}".format(username))
        return 
    elif username in HD:
        c = C_HD
        group = 'HD'
    elif username in LD:
        group = 'LD'
        c = C_LD
    
    # for df, phase in zip([df_PRE, df_COND, df_POST], ["PRE", "COND", "POST"]):
    for df, phase in zip([df_PRE], ["PRE"]):
        # print("##### {} #####".format(phase))
        genres_EM, genres_all = search_EM_genres(df)
        artists_EM, artists_all = search_EM_artists(df, genres_EM)
        if phase == 'PRE':
            c['PRE'].update(Counter((artists_EM)))
        elif phase == 'COND':
            c['COND'].update(Counter((artists_EM)))
        elif phase == 'POST':
            c['POST'].update(Counter((artists_EM)))

    



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

    rows = []
    rows_diff =  []
    C_HD = {'PRE':Counter(), 'COND':Counter(), 'POST':Counter()}
    C_LD = {'PRE':Counter(), 'COND':Counter(), 'POST':Counter()}

    if username:
        # rows = run(username, rows)
        # rows_diff = run_diff(username, rows_diff)
        run_qual(username, C_HD, C_LD)

    elif input_file:
        infile = open(input_file, 'r')
        lines = infile.readlines()
        infile.close()

        for line in lines:
            print("-------------")
            username = line.strip()
            # rows = run(username, rows)
            # rows_diff = run_diff(username, rows_diff)
            run_qual(username, C_HD, C_LD)


    # with open(OUTFILE_1, 'w+') as outf:
    #     _writer = csv.writer(outf)
    #     _writer.writerow(HEADER)
    #     for row in rows:
    #         _writer.writerow(row)
    # with open(OUTFILE_2, 'w+') as outf:
    #     _writer = csv.writer(outf)
    #     _writer.writerow(HEADER_DIFF)
    #     for row in rows_diff:
    #         _writer.writerow(row)





    count_HD = sum([C_HD['PRE'], C_HD['COND'], C_HD['POST']], Counter())
    count_LD = sum([C_LD['PRE'], C_LD['COND'], C_LD['POST']], Counter())
    counts = sum([count_LD, count_HD], Counter())

    # fig, axs = plt.subplots()
    # wc = WordCloud(background_color="white",
    #                width=1000,height=1000,
    #                max_words=50,
    #                relative_scaling=0.5,
    #                normalize_plurals=False).generate_from_frequencies(
    #                counts)
    # axs.imshow(wc)
    # plt.show()


    # map_count = {}
    # for k,v in counts.items():
    #     new_k = EM_MAP[k]
    #     if new_k not in map_count:
    #         map_count[new_k] = 0
    #     map_count[new_k] += v

    print(counts)
    print(sum(counts.values()))
    # print(map_count)




    # import shifterator as sh
    # jsd_shift = sh.JSDivergenceShift(type2freq_1=C_HD["PRE"],
    #                                  type2freq_2=C_LD["PRE"],
    #                                  weight_1=0.5,
    #                                  weight_2=0.5,
    #                                  base=2,
    #                                  alpha=1.5)

    # jsd_shift.get_shift_graph(system_names = ['HD', 'LD'], top_n=20, title='PRE')


    # proportion_shift = sh.ProportionShift(type2freq_1=C_HD["PRE"],
    #                                       type2freq_2=C_LD["PRE"])
    # proportion_shift.get_shift_graph(system_names = ['HD.', 'LD'])

    # entropy_shift = sh.EntropyShift(type2freq_1=C_HD["PRE"],
    #                                 type2freq_2=C_HD["COND"],
    #                                 base=2)
    # entropy_shift.get_shift_graph(system_names = ['HD_PRE', 'HD_COND'])

    # entropy_shift = sh.EntropyShift(type2freq_1=C_HD["COND"],
    #                                 type2freq_2=C_HD["POST"],
    #                                 base=2)
    # entropy_shift.get_shift_graph(system_names = ['HD_COND', 'HD_POST'])

    # entropy_shift = sh.EntropyShift(type2freq_1=C_HD["PRE"],
    #                                 type2freq_2=C_HD["POST"],
    #                                 base=2)
    # entropy_shift.get_shift_graph(system_names = ['HD_PRE', 'HD_POST'])


    # entropy_shift = sh.EntropyShift(type2freq_1=C_LD["PRE"],
    #                                 type2freq_2=C_LD["COND"],
    #                                 base=2)
    # entropy_shift.get_shift_graph(system_names = ['LD_PRE', 'LD_COND'])

    # entropy_shift = sh.EntropyShift(type2freq_1=C_LD["COND"],
    #                                 type2freq_2=C_LD["POST"],
    #                                 base=2)
    # entropy_shift.get_shift_graph(system_names = ['LD_COND', 'LD_POST'])


    # entropy_shift = sh.EntropyShift(type2freq_1=C_LD["PRE"],
    #                                 type2freq_2=C_LD["POST"],
    #                                 base=2)
    # entropy_shift.get_shift_graph(system_names = ['LD_PRE', 'LD_POST'])



    # fig, axs = plt.subplots(2,3, sharey=True, sharex=True)

    # wc = WordCloud(background_color="white",
    #                    width=1000,height=1000,
    #                    max_words=50,
    #                    relative_scaling=0.5,
    #                    normalize_plurals=False).generate_from_frequencies(
    #                    C_HD["PRE"])#-C_HD["COND"]-C_HD["POST"])
    # axs[0,0].imshow(wc)
    # wc = WordCloud(background_color="white",
    #                    width=1000,height=1000,
    #                    max_words=50,
    #                    relative_scaling=0.5,
    #                    normalize_plurals=False).generate_from_frequencies(
    #                    C_HD["COND"])#-C_HD["PRE"]-C_HD["POST"])
    # axs[0,1].imshow(wc)
    # wc = WordCloud(background_color="white",
    #                    width=1000,height=1000,
    #                    max_words=50,
    #                    relative_scaling=0.5,
    #                    normalize_plurals=False).generate_from_frequencies(
    #                    C_HD["POST"])#-C_HD["COND"]-C_HD["PRE"])
    # axs[0,2].imshow(wc)
    # wc = WordCloud(background_color="white",
    #                    width=1000,height=1000,
    #                    max_words=50,
    #                    relative_scaling=0.5,
    #                    normalize_plurals=False).generate_from_frequencies(
    #                    C_LD["PRE"])#-C_LD["COND"]-C_LD["POST"])
    # axs[1,0].imshow(wc)
    # wc = WordCloud(background_color="white",
    #                    width=1000,height=1000,
    #                    max_words=50,
    #                    relative_scaling=0.5,
    #                    normalize_plurals=False).generate_from_frequencies(
    #                    C_LD["COND"])#-C_LD["PRE"]-C_LD["POST"])
    # axs[1,1].imshow(wc)
    # wc = WordCloud(background_color="white",
    #                    width=1000,height=1000,
    #                    max_words=50,
    #                    relative_scaling=0.5,
    #                    normalize_plurals=False).generate_from_frequencies(
    #                    C_LD["POST"])#-C_LD["COND"]-C_LD["PRE"])
    # axs[1,2].imshow(wc)
    # plt.show()


    # l1 = (C_HD["PRE"]-C_HD["COND"]-C_HD["POST"]).values()
    # print(l1)
    # print(len(l1))
    # print(gini([*l1]))
    # print(entropy([x/sum(l1) for x in [*l1]]))
    # l1 = (C_HD["COND"]-C_HD["PRE"]-C_HD["POST"]).values()
    # print(l1)
    # print(len(l1))
    # print(gini([*l1]))
    # print(entropy([x/sum(l1) for x in [*l1]]))
    # l1 = (C_HD["POST"]-C_HD["COND"]-C_HD["PRE"]).values()
    # print(l1)
    # print(len(l1))
    # print(gini([*l1]))
    # print(entropy([x/sum(l1) for x in [*l1]]))
    # l1 = (C_LD["PRE"]-C_LD["COND"]-C_LD["POST"]).values()
    # print(l1)
    # print(len(l1))
    # print(gini([*l1]))
    # print(entropy([x/sum(l1) for x in [*l1]]))
    # l1 = (C_LD["COND"]-C_LD["PRE"]-C_LD["POST"]).values()
    # print(l1)
    # print(len(l1))
    # print(gini([*l1]))
    # print(entropy([x/sum(l1) for x in [*l1]]))
    # l1 = (C_LD["POST"]-C_LD["COND"]-C_LD["PRE"]).values()
    # print(l1)
    # print(len(l1))
    # print(gini([*l1]))
    # print(entropy([x/sum(l1) for x in [*l1]]))
