#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import scipy.stats as stats

from scipy.interpolate import UnivariateSpline
from pingouin import ttest, wilcoxon, mwu

ATT_FOLDER = "../data/attitudes"
LS_FOLDER = "../data/ls"

CONTEXTS = ["Relaxing", "Commuting", "Partying", "Running","Shopping",
            "Sleeping", "Studying", "Working"]

CONTEXTS_a = ["Relaxing", "Sleeping", "Studying", "Working"]
CONTEXTS_b = ["Commuting", "Partying", "Running", "Shopping"]
TRACK_FEATS = ["Tempo", "Danceability", "Acousticness", "Instrumentalness"]
ARTIST_FEATS = ["Gender", "Skin", "Origin", "Age"]
ROUNDS =  ["00", "01", "02", "03", "04", "10"]
ROUNDS_LAB = ['Pre', 'Week 1', "Week 2", "Week 3", "Week 4" "Post"]
GROUPS = ["HD", "LD"]
SESSION1 = [str(x).zfill(2) for x in range(1,6)]
SESSION2 = [str(x).zfill(2) for x in range(6,11)]
SESSION3 = [str(x).zfill(2) for x in range(11,16)]
SESSION4 = [str(x).zfill(2) for x in range(16,21)]
SESSIONS = [SESSION1, SESSION2, SESSION3, SESSION4]

YT_HD = [36,23,17,16,29,11,10,6,12,4,13,0,21,8,14,1,27,9,6,3]
YT_LD = [11,11,4,18,2,1,19,1,1,0,0,14,7,1,4,4,0,3,9,1]


def import_data(tdata):
    """
    """
    dfs = []
    df_join = pd.DataFrame()    
    for att_round in ROUNDS:
        for group in GROUPS:
            group_infile = "{}_{}_{}.csv".format(group, att_round, tdata)
            infile = os.path.join(ATT_FOLDER, att_round, group_infile)
            df = pd.read_csv(infile)
            df = df.assign(group=[group for x in df.index],
                           att_round=[att_round for x in df.index])
            dfs.append(df)
    df_join = pd.concat(dfs, ignore_index=True)

    return df_join


def import_ls_data():
    """
    """
    dfs = []
    df_join = pd.DataFrame()  
    for group in GROUPS:
        LS_FOLDER_G = os.path.join(LS_FOLDER, group)
        for infile in sorted(os.listdir(LS_FOLDER_G)):
            base, ext = infile.split(".")
            _, _, session = base.split("_")

            infile = os.path.join(LS_FOLDER_G, infile)
            df = pd.read_csv(infile)
            df = df.assign(group=[group for x in df.index],
                           session=[session for x in df.index])
            dfs.append(df)
        
        df_join = pd.concat(dfs, ignore_index=True)

    return df_join


def scale_ls_data(df):
    """
    """
    # Scale Data
    df.loc[df["familiarity"] == 2, "familiarity"] = -1
    df.loc[df["familiarity"] == 3, "familiarity"] = 0
    df['like'] -= 3
    df.loc[df["playlist"] == 2, "playlist"] = 0

    return df


def import_whitelist(list_type):
    """
    """
    whitelist_pid = []
    whitelist_uname = []
    
    if list_type == 'full':
        infile = "../data/PID_full.csv"
    elif list_type == 'inc':
        infile = "../data/PID_inc.csv"
    else:
        return whitelist_id, whitelist_uname

    with open(infile, 'r') as inf:   
        _reader = csv.reader(inf, delimiter='\t')
        for row in _reader:
            whitelist_pid.append(row[1])
            whitelist_uname.append(row[0])

    return whitelist_pid, whitelist_uname


def filter_dataframe(df, list_type, list_ent):
    """
    """
    print("Original Dataframe Shape: {}".format(df.shape))
    wlist_id, wlist_uname = import_whitelist(list_type)
    if list_ent == 'pid':
        df = df[df.PROLIFIC_PID.isin(wlist_id)]
        print("Filtered Dataframe Shape: {}".format(df.shape))
    elif list_ent == 'uname':
        df = df[df.username.isin(wlist_uname)]
        print("Filtered Dataframe Shape: {}".format(df.shape))        
    return df


def test_significance(l1, l2):
    """
    """
    print(np.median(l1), np.std(l1))
    print(np.median(l2), np.std(l2))
    print(mwu(l1, l2, alternative='less'))
    print()
    




if __name__ == "__main__":

    df_join_ls = import_ls_data()
    df_join_ls = scale_ls_data(df_join_ls)
    df_join_ls = filter_dataframe(df_join_ls, 'inc', 'pid')

    df_join_ls_HD = df_join_ls[df_join_ls.group == 'HD']
    df_join_ls_LD = df_join_ls[df_join_ls.group == 'LD']

    fsize = 20

    # Playlist Access VS Interactions
    p_HD = df_join_ls_HD.groupby('session').playlist.sum().tolist()
    p_LD = df_join_ls_LD.groupby('session').playlist.sum().tolist()

    fig, ax = plt.subplots()
    ax.bar(np.arange(20), p_HD, width=0.3, label='HD', alpha=0.4, color='r')
    ax.bar(np.arange(20)-0.3, p_LD, width=0.3, label='LD', hatch="-", alpha=0.4, color='g')
    ax.scatter(np.arange(20), YT_HD, color='r', marker='>', facecolors='none', s=50, label='Playlist Interaction (HD)')
    ax.scatter(np.arange(20)-0.3, YT_LD, color='g', marker='x', s=50, label='Playlist Interaction (LD)')

    a, b = np.polyfit(np.arange(20), p_HD, 1)
    ax.plot(np.arange(20), a*np.arange(20)+b,color='r', linestyle='--')  
    a, b = np.polyfit(np.arange(20), p_LD, 1)
    ax.plot(np.arange(20)-0.3, a*np.arange(20)+b,color='g', linestyle='--')  

    ax.set_xticklabels(np.arange(1,21), fontsize = fsize)
    ax.set_xticks(np.arange(20)-0.15)
    ax.set_xlabel('Session', fontsize = fsize)
    ax.set_ylabel('Sum of Playlists Access', fontsize = fsize)
    # ax.set_title('Playlist Access VS Playlist Interaction', fontsize = fsize)
    plt.legend(fontsize = fsize)
    plt.grid()
    plt.show()

    # T-test
    test_significance(p_HD, p_LD) #1
    test_significance(YT_HD, YT_LD) #2 
    d_HD = [x-y for x,y in zip(p_HD,YT_HD)]
    d_LD = [x-y for x,y in zip(p_LD,YT_LD)]
    test_significance(d_HD, d_LD) #3 

    print(stats.pearsonr(p_HD, YT_HD))
    print(stats.pearsonr(p_LD, YT_LD))


    # Like Ratings VS Playlist Interactions
    l_HD = df_join_ls_HD.groupby('session').like.sum().tolist()
    l_LD = df_join_ls_LD.groupby('session').like.sum().tolist()

    fig, ax = plt.subplots()
    ax.bar(np.arange(20), l_HD, width=0.3, label='HD', alpha=0.4, color='r')
    ax.bar(np.arange(20)-0.3, l_LD, width=0.3, label='LD',hatch="-", alpha=0.4, color='g')
    ax.scatter(np.arange(20), YT_HD, color='r', marker='>', facecolors='none', s=50, label='Playlist Interaction (HD)')
    ax.scatter(np.arange(20)-0.3, YT_LD, color='g', marker='x',s=50, label='Playlist Interaction (LD)')

    a, b = np.polyfit(np.arange(20), l_HD, 1)
    ax.plot(np.arange(20), a*np.arange(20)+b,color='r', linestyle='--')  
    a, b = np.polyfit(np.arange(20),l_LD, 1)
    ax.plot(np.arange(20)-0.3, a*np.arange(20)+b,color='g', linestyle='--')  

    ax.set_xticklabels(np.arange(1,21), fontsize = fsize)
    ax.set_xticks(np.arange(20)-0.15)
    ax.set_ylabel('Sum of Like Ratings', fontsize = fsize)
    ax.set_xlabel('Session', fontsize = fsize)
    # ax.set_title('Like Ratings VS Playlist Interaction', fontsize = fsize)
    plt.legend(fontsize = fsize)
    plt.grid()
    plt.show()

    # T-test
    test_significance(l_HD, l_LD) #4 

    print(stats.pearsonr(l_HD, p_HD))
    print(stats.pearsonr(l_LD, p_LD)) 


    # Like Ratings VS Playlist Access
    HD_like_sum_p = df_join_ls_HD[df_join_ls_HD.playlist==1].groupby('session').like.sum()
    HD_like_sum_n = df_join_ls_HD[df_join_ls_HD.playlist==0].groupby('session').like.sum()
    HD_bottom = []
    for p,n in zip(HD_like_sum_p, HD_like_sum_n):
        if np.sign(p) == np.sign(n):
            HD_bottom.append(p)
        else:
            HD_bottom.append(0)

    LD_like_sum_p = df_join_ls_LD[df_join_ls_LD.playlist==1].groupby('session').like.sum()
    LD_like_sum_n = df_join_ls_LD[df_join_ls_LD.playlist==0].groupby('session').like.sum()
    LD_bottom = []
    for p,n in zip(LD_like_sum_p, LD_like_sum_n):
        if np.sign(p) == np.sign(n):
            LD_bottom.append(p)
        else:
            LD_bottom.append(0)

    fig, ax = plt.subplots()
    ax.bar(np.arange(20), HD_like_sum_p, width=0.3, label='HD, playlist', alpha=0.5, color='r')
    ax.bar(np.arange(20), HD_like_sum_n, width=0.3, label='HD, no playlist', alpha=0.7, color='r', bottom=HD_bottom)
    ax.bar(np.arange(20)-0.3, LD_like_sum_p, width=0.3, label='LD, playlist', alpha=0.5,hatch="-", color='g')
    ax.bar(np.arange(20)-0.3, LD_like_sum_n, width=0.3, label='LD, no playlist', alpha=0.7,hatch="-", color='g', bottom=LD_bottom)
    ax.set_xticklabels(np.arange(1,21), fontsize = fsize)
    ax.set_xticks(np.arange(20)-0.15)
    ax.set_ylabel('Sum of Like Ratings', fontsize = fsize)
    ax.set_xlabel('Session', fontsize = fsize)
    # ax.set_title('Like Ratings VS Playlist Access', fontsize = fsize)
    plt.legend(fontsize = fsize)
    plt.grid()
    plt.show()


    # T-test
    test_significance(HD_like_sum_p, LD_like_sum_p) #5 
    test_significance(HD_like_sum_n, LD_like_sum_n) #6 

    

    # Familiarity 
    fig, ax = plt.subplots()

    f_HD = df_join_ls_HD.groupby('session').familiarity.sum().tolist()
    f_LD = df_join_ls_LD.groupby('session').familiarity.sum().tolist()

    ax.bar(np.arange(20), f_HD, width=0.3, label='HD', alpha=0.5, color='r')
    ax.bar(np.arange(20)-0.3, f_LD , width=0.3, label='LD', alpha=0.5, hatch="-", color='g')


    # a, b = np.polyfit(np.arange(20), f_HD, 1)
    # ax.plot(np.arange(20), a*np.arange(20)+b,color='r', linestyle='--')  
    # a, b = np.polyfit(np.arange(20), f_LD, 1)
    # ax.plot(np.arange(20)-0.3, a*np.arange(20)+b,color='g', linestyle='--')  
    
    ax.set_xticklabels(np.arange(1,21))
    ax.set_xticks(np.arange(20)-0.15)
    ax.set_ylabel('Sum of Familiarity Ratings', fontsize = fsize)
    ax.set_xlabel('Session', fontsize = fsize)
    # ax.set_title('Familiarity', fontsize = fsize)
    plt.legend(fontsize = fsize)
    plt.grid()
    plt.show()

    test_significance(f_HD, f_LD) #7


    print(stats.pearsonr(l_HD, f_HD))
    print(stats.pearsonr(l_LD, f_LD))



    # Familiarity VS Like Ratings
    HD_like_sum_p = df_join_ls_HD[df_join_ls_HD.like>0].groupby('session').familiarity.sum()
    HD_like_sum_n = df_join_ls_HD[df_join_ls_HD.like<0].groupby('session').familiarity.sum()
    HD_bottom = []
    for p,n in zip(HD_like_sum_p, HD_like_sum_n):
        if np.sign(p) == np.sign(n):
            HD_bottom.append(p)
        else:
            HD_bottom.append(0)

    LD_like_sum_p = df_join_ls_LD[df_join_ls_LD.like>0].groupby('session').familiarity.sum()
    LD_like_sum_n = df_join_ls_LD[df_join_ls_LD.like<0].groupby('session').familiarity.sum()
    LD_bottom = []
    for p,n in zip(LD_like_sum_p, LD_like_sum_n):
        if np.sign(p) == np.sign(n):
            LD_bottom.append(p)
        else:
            LD_bottom.append(0)

    fig, ax = plt.subplots()
    ax.bar(np.arange(20), HD_like_sum_p, width=0.3, label='HD, like', alpha=0.5, color='r')
    ax.bar(np.arange(20), HD_like_sum_n, width=0.3, label='HD, dislike', alpha=0.7, color='r', bottom=HD_bottom)
    ax.bar(np.arange(20)-0.3, LD_like_sum_p, width=0.3, label='LD, like', alpha=0.5,hatch="-", color='g')
    ax.bar(np.arange(20)-0.3, LD_like_sum_n, width=0.3, label='LD, dislike', alpha=0.7,hatch="-", color='g', bottom=LD_bottom)
    ax.set_xticklabels(np.arange(1,21))
    ax.set_xticks(np.arange(20)-0.15)
    ax.set_ylabel('Sum of Familiarity Ratings', fontsize = fsize)
    ax.set_xlabel('Session', fontsize = fsize)
    # ax.set_title('Familiarity VS Like Ratings', fontsize = fsize)
    plt.legend(fontsize = fsize)
    plt.grid()
    plt.show()



    # T-test
    test_significance(HD_like_sum_p, LD_like_sum_p) #8
    test_significance(HD_like_sum_n, LD_like_sum_n) #9
