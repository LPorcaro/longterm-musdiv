#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta
from pylistenbrainz.errors import ListenBrainzAPIException
from collections import Counter

WIKI_GENRES = "../data/input/wikipedia_EM_genres.csv"
FEAT_DIR = "../data/listenbrainz/feat"
INFO_DIR = "../data/listenbrainz/info"
# STATS_DIR = "../data/listenbrainz/stats"

end = datetime.strptime("20220216", "%Y%m%d")

def search_em_genres(df):
    """
    """
    EM_genres = []
    with open(WIKI_GENRES) as inf:
        _reader = csv.reader(inf)
        for row in _reader:
            EM_genres.append(row[0].lower())
    EM_genres.append("electronic")

    genres_list = [x.split(",") for x in df.genres.values if pd.isnull(x) == False and x != 'Empty']
    genres_list = [y for l in genres_list for y in l]

    genres_found = []
    for genre in EM_genres:
        if genre in genres_list:
            genres_found.append(genre)
        else:
            stringmatch = [x for x in genres_list if genre in x]
            if stringmatch:
                genres_found += stringmatch

    return genres_found, genres_list

def write_stats(df, outfile):
    """
    """
    file = open(outfile, 'w+')
    if df.empty:
        file.close()
    else:
        df.genres.fillna('Empty', inplace=True)
        genres_found, genres_list = search_em_genres(df)

        if genres_found:
            track_found = df[df.genres.str.contains('|'.join(genres_found), regex=True)].track_name.values
            artist_found = df[df.genres.str.contains('|'.join(genres_found), regex=True)].artist_name.values
        else:
            track_found = []
            artist_found = []
    
        percentage = len(track_found)*100/len(df.index)
        percentage = round(percentage, 3)

        artist_list = [x.split(",") for x in df.artist_name.values if pd.isnull(x) == False]
        artist_list = set([y for l in artist_list for y in l])
        isrc_co_list = [x[:2] for x in df.ISRC.values if pd.isnull(x) == False]
        isrc_reg_list = [x[2:5] for x in df.ISRC.values if pd.isnull(x) == False]
        isrc_year_list = [x[5:7] for x in df.ISRC.values if pd.isnull(x) == False]

        file.write("Electronic artists found: {}\n".format(", ".join(set(artist_found))))
        file.write("Electronic genres found: {}\n".format(", ".join(set(genres_found))))
        file.write("Electronic music logs found: {:.2f}% \n\n".format(percentage))
        file.write("Top genres\n")
        for c, el in enumerate(Counter(genres_list).most_common(5)):
            file.write("\t{}: {} ({:.2f}%)\n".format(c+1, el[0], el[1]*100/len(genres_list)))
        file.write("Top artists\n")
        for c, el in enumerate(Counter(artist_list).most_common(5)):
            file.write("\t{}: {} ({:.2f}%)\n".format(c+1, el[0], el[1]*100/len(artist_list)))
        file.write("Top country\n")
        for c, el in enumerate(Counter(isrc_co_list).most_common(5)):
            file.write("\t{}: {} ({:.2f}%)\n".format(c+1, el[0], el[1]*100/len(isrc_co_list)))
        file.write("Top record\n")
        for c, el in enumerate(Counter(isrc_reg_list).most_common(5)):
            file.write("\t{}: {} ({:.2f}%)\n".format(c+1, el[0], el[1]*100/len(isrc_reg_list)))
        file.write("Top year\n")
        for c, el in enumerate(Counter(isrc_year_list).most_common(5)):
            file.write("\t{}: {} ({:.2f}%)\n".format(c+1, el[0], el[1]*100/len(isrc_year_list)))

        file.write("\nFeature statistics\n")
        file.write(df.describe().to_string())

    return track_found, genres_found, artist_found

def analyze_user_temporal(username, temporal_feat):
    """
    """
    (Day, N, EM, P_mean, P_std, A_mean, A_std,
            D_mean, D_std, I_mean, I_std, T_mean, T_std) = temporal_feat

    fig, ax = plt.subplots(3, 2, sharex=True)
    fmt = '--'
    n = len(Day)
    r = np.arange(n)
    width = 0.25

    # ax[0, 0].plot(Day, N)
    # ax[0, 0].plot(Day, EM, linestyle='dashed')

    ax[0, 0].bar(r, N, color = 'b',
            width = width, edgecolor = 'black',
            label='Track Count')
    ax[0, 0].bar(r + width, EM, color = 'g',
            width = width, edgecolor = 'black',
            label='EM Track')
      
    # plt.grid(linestyle='--')
    ax[0, 0].set_xticks(r + width/2)
    ax[0, 0].set_xticklabels(Day)
    ax[0, 0].legend()

    ax[0, 0].set_title('Tracks count VS Electronic Music')

    ax[0, 1].errorbar(Day, P_mean, P_std, fmt=fmt)
    ax[0, 1].set_ylim([0, 100])
    ax[0, 1].set_title('Popularity')

    ax[1, 0].errorbar(Day, A_mean, A_std, color='#0504aa', fmt=fmt)
    ax[1, 0].set_ylim([0, 1.5])
    ax[1, 0].set_title('Acousticness')

    ax[1, 1].errorbar(Day, D_mean, D_std, color='#baba07',fmt=fmt)
    ax[1, 1].set_ylim([0, 1.5])
    ax[1, 1].set_title('Danceability')

    ax[2, 0].errorbar(Day, I_mean, I_std, color='#fb0a66',fmt=fmt)
    ax[2, 0].set_ylim([0,1.5])
    ax[2, 0].set_title('Instrumentalness')

    ax[2, 1].errorbar(Day, T_mean, T_std, fmt=fmt)
    ax[2, 1].set_ylim([0,200])
    ax[2, 1].set_title('BPM')

    fig.suptitle(username)

    plt.show()


def analyze_logs(username, date, out_dir):
    """
    """
    print("User '{}': Analyzing...".format(username))

    inf_info = "{}_{}_info.csv".format(username, date)
    inf_info = os.path.join(INFO_DIR, username, inf_info)
    if not os.path.exists(inf_info):
        print("User '{}': INFO file not found {} ".format(username, inf_info))
        return
    df_info = pd.read_csv(inf_info, delimiter="\t")

    inf_feat = "{}_{}_feat.csv".format(username, date)
    inf_feat = os.path.join(FEAT_DIR, username, inf_feat)
    if not os.path.exists(inf_feat):
        print("User '{}': INFO file not found {} ".format(username, inf_feat))
        return
    df_feat = pd.read_csv(inf_feat, delimiter="\t")

    if df_info.empty or df_feat.empty:
        print ("User '{}': No listening logs found!".format(username))
        return

    STATS_DIR = os.path.join(out_dir, username)
    if not os.path.exists(STATS_DIR):
        os.makedirs(STATS_DIR)    
    
    start = datetime.strptime(date, "%Y%m%d")

    (Day, N, EM, A_EM, G_EM, P_mean, P_std, A_mean, A_std,
        D_mean, D_std, I_mean, I_std, T_mean, T_std) = ([], [], [], [], [], [],
        [], [], [], [], [], [], [], [], [])

    while start > end:
        start_str = start.strftime("%Y-%m-%d")
        df = pd.DataFrame()
        
        mask = df_info.listened_at.str.startswith(start_str)
        df = df_feat.loc[mask[mask].index]

        outfile = "{}-{}.txt".format(username, start_str)
        outfile = os.path.join(STATS_DIR, outfile)

        start_str = start_str[5:]
        if df.empty:
            Day.append(start_str)
            N.append(0)
            EM.append(0)
            P_mean.append(0)
            P_std.append(0)
            A_mean.append(0)
            A_std.append(0)
            D_mean.append(0)
            D_std.append(0)
            I_mean.append(0)
            I_std.append(0)
            T_mean.append(0)
            T_std.append(0)
            print ("{}: No listening logs found!".format(start_str))
        else:
            track_found, genres_found, artist_found = write_stats(df, outfile)
            Day.append(start_str)
            N.append(len(df.index))
            EM.append(len(track_found))
            A_EM.extend(artist_found)
            G_EM.extend(genres_found)
            P_mean.append(df.popularity.mean())
            P_std.append(df.popularity.std())
            A_mean.append(df.acousticness.mean())
            A_std.append(df.acousticness.std())
            D_mean.append(df.danceability.mean())
            D_std.append(df.danceability.std())
            I_mean.append(df.instrumentalness.mean())
            I_std.append(df.instrumentalness.std())
            T_mean.append(df.tempo.mean())
            T_std.append(df.tempo.std())
            print("{}: Analyzed {} listen events".format(
                start_str, len(df.index)))   

        start -= timedelta(hours=24)

    Day.reverse()
    N.reverse()
    EM.reverse()
    P_mean.reverse()
    P_std.reverse()
    A_mean.reverse()
    A_std.reverse()
    D_mean.reverse()
    D_std.reverse()
    I_mean.reverse()
    I_std.reverse()
    T_mean.reverse()
    T_std.reverse()

    temporal_feat = (Day, N, EM, P_mean, P_std, A_mean, A_std,
        D_mean, D_std, I_mean, I_std, T_mean, T_std)

    analyze_user_temporal(username, temporal_feat)

    print(len(set(A_EM)), len(set(G_EM)))


def arg_parser():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", type=str, dest='username',
                        help="ListenBrainz Username")
    parser.add_argument("-d", "--date", type=str, dest='date',
                        help="Log Date")
    parser.add_argument("-i", "--input", type=str, dest='input_file',
                        help="Input file with ListenBrainz usernames")
    parser.add_argument("-o", "--output", type=str, dest='out_dir',
                        help="Output for JSON files")
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = arg_parser()

    date = args.date
    username = args.username
    input_file = args.input_file
    out_dir = args.out_dir

    if username:
        analyze_logs(username, date, out_dir)

    elif input_file:
        infile = open(input_file, 'r')
        lines = infile.readlines()
        infile.close()

        for line in lines:
            try:
                username = line.strip()
                analyze_logs(username, date, out_dir)
            except ListenBrainzAPIException:
                print("Problems analyzing logs: {}".format(username))
