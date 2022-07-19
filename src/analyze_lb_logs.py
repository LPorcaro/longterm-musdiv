#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np

from tabulate import tabulate
from datetime import datetime, timedelta
from collections import Counter, OrderedDict
from itertools import combinations
from scipy.stats import pearsonr, norm

INFILE_LOGS = "../data/listenbrainz/results/logs_analysis.csv"
INFILE_DIFF = "../data/listenbrainz/results/logs_diff_analysis.csv"

ATT_FOLDER = "../data/attitudes"
LS_FOLDER = "../data/ls"

CONTEXTS = ["Relaxing", "Commuting", "Partying", "Running","Shopping",
            "Sleeping", "Studying", "Working"]

CONTEXTS_a = ["Relaxing", "Sleeping", "Studying", "Working"]
CONTEXTS_b = ["Commuting", "Partying", "Running", "Shopping"]
TRACK_FEATS = ["Tempo", "Danceability", "Acousticness", "Instrumentalness"]
ARTIST_FEATS = ["Gender", "Skin", "Origin", "Age"]
ROUNDS =  ["00", "01", "02", "03", "04",]# "10"]
ROUNDS_LAB = ['Pre', 'Week 1', "Week 2", "Week 3", "Week 4"]#, "Post"]
GROUPS = ["HD", "LD"]
SESSION1 = [str(x).zfill(2) for x in range(1,6)]
SESSION2 = [str(x).zfill(2) for x in range(6,11)]
SESSION3 = [str(x).zfill(2) for x in range(11,16)]
SESSION4 = [str(x).zfill(2) for x in range(16,21)]
SESSIONS = [SESSION1, SESSION2, SESSION3, SESSION4]


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


if __name__ == "__main__":


    df_logs = pd.read_csv(INFILE_LOGS)

    df_logs_HD = df_logs[df_logs.group == 'HD']
    df_logs_LD = df_logs[df_logs.group == 'LD']

    df_logs_PRE = df_logs[df_logs.phase == 'PRE']
    df_logs_COND = df_logs[df_logs.phase == 'COND']
    df_logs_POST = df_logs[df_logs.phase == 'POST']

    pre_HD = df_logs_HD[df_logs_HD.phase == 'PRE']
    cond_HD = df_logs_HD[df_logs_HD.phase == 'COND']
    post_HD = df_logs_HD[df_logs_HD.phase == 'POST']

    pre_LD = df_logs_LD[df_logs_LD.phase == 'PRE']
    cond_LD = df_logs_LD[df_logs_LD.phase == 'COND']
    post_LD = df_logs_LD[df_logs_LD.phase == 'POST']

    df_diff = pd.read_csv(INFILE_DIFF)

    df_diff_HD = df_diff[df_diff.group == 'HD']
    df_diff_LD = df_diff[df_diff.group == 'LD']

    df_diff_COND = df_diff[df_diff.phase == 'COND']
    df_diff_POST = df_diff[df_diff.phase == 'POST']

    cond_diff_HD = df_diff_HD[df_diff_HD.phase == 'COND']
    post_diff_HD = df_diff_HD[df_diff_HD.phase == 'POST']

    cond_diff_LD = df_diff_LD[df_diff_LD.phase == 'COND']
    post_diff_LD = df_diff_LD[df_diff_LD.phase == 'POST']



    # atts = ["genres_unique","genres_count","EM_genres_unique","EM_genres_count", 
    #         "artists_unique","artists_count","EM_artists_unique",
    #         "EM_artists_count","tracks_unique","tracks_count",
    #         "EM_tracks_unique","EM_tracks_count"]


    # atts = ["EM_genres_unique_p", "EM_genres_count_p", "EM_genres_top_head_p",
    #         "EM_artists_unique_p", "EM_artists_count_p", "EM_artists_top_head_p",
    #         "EM_tracks_unique_p", "EM_tracks_count_p","EM_tracks_top_head_p"]

    # atts = ["gini_genres", "gini_artists", "gini_tracks",
    #         "gini_EM_genres","gini_EM_artists","gini_EM_tracks"]

    # atts = ["genres_unique","artists_unique","tracks_unique",
    #         "EM_genres_unique", "EM_artists_unique", "EM_tracks_unique"]

    atts = ["genres_count", "artists_count", "tracks_count",
            "EM_genres_count", "EM_artists_count", "EM_tracks_count"]

    # atts = ["nov genres","nov artists","nov tracks",
    #         "nov EM genres","nov EM artists","nov EM tracks"]


    # ############################## DID ##############################   
    # for att in atts:
    #     print()
    #     y11 = np.mean((np.where(pre_LD[att].values>0, (pre_LD[att].values), 0)))
    #     y12 = np.mean((np.where(post_LD[att].values>0, (post_LD[att].values), 0)))
    #     y21 = np.mean((np.where(pre_HD[att].values>0, (pre_HD[att].values), 0)))
    #     y22 = np.mean((np.where(post_HD[att].values>0, (post_HD[att].values), 0)))

    #     data = ['PRE', y21, y11, y21-y11],['POST',y22, y12, y22-y12],['Change',y22-y21, y12-y11, (y22-y12)-(y21-y11)]
    #     print(tabulate(data, headers=[att,'Treat.', 'Cont.', 'Diff.'], tablefmt="github"))


    # ############################## SCATTER ##############################
    # for att in atts:
    #     fig, axs = plt.subplots(2,3, sharex=True, sharey=True)

    #     # pre_gu_HD =  np.where(pre_HD[att].values>0, np.log(pre_HD[att].values), 0)
    #     # cond_gu_HD = np.where(cond_HD[att].values>0, np.log(cond_HD[att].values), 0)
    #     # post_gu_HD = np.where(post_HD[att].values>0, np.log(post_HD[att].values), 0)
    #     # pre_gu_LD = np.where(pre_LD[att].values>0, np.log(pre_LD[att].values), 0)
    #     # cond_gu_LD = np.where(cond_LD[att].values>0, np.log(cond_LD[att].values), 0)
    #     # post_gu_LD = np.where(post_LD[att].values>0, np.log(post_LD[att].values), 0)

    #     pre_gu_HD =  np.where(pre_HD[att].values>0, (pre_HD[att].values), 0)
    #     cond_gu_HD = np.where(cond_HD[att].values>0, (cond_HD[att].values), 0)
    #     post_gu_HD = np.where(post_HD[att].values>0, (post_HD[att].values), 0)
    #     pre_gu_LD = np.where(pre_LD[att].values>0, (pre_LD[att].values), 0)
    #     cond_gu_LD = np.where(cond_LD[att].values>0, (cond_LD[att].values), 0)
    #     post_gu_LD = np.where(post_LD[att].values>0, (post_LD[att].values), 0)

    #     xm = np.min([np.min(x) for x in [pre_gu_HD,cond_gu_HD,post_gu_HD,pre_gu_LD, cond_gu_LD,post_gu_LD]])
    #     xM = np.max([np.max(x) for x in [pre_gu_HD,cond_gu_HD,post_gu_HD,pre_gu_LD, cond_gu_LD,post_gu_LD]])+1

    #     axs[0,0].scatter(pre_gu_HD,cond_gu_HD, label=pearsonr(pre_gu_HD,cond_gu_HD))
    #     axs[0,0].plot(np.arange(xm,xM), np.arange(xm,xM), c='r', linestyle='--')
    #     axs[0,0].legend(loc="upper right")
    #     axs[0,1].scatter(pre_gu_HD,post_gu_HD, label=pearsonr(pre_gu_HD,post_gu_HD))
    #     axs[0,1].plot(np.arange(xm,xM), np.arange(xm,xM), c='r', linestyle='--')
    #     axs[0,1].legend(loc="upper right")
    #     axs[0,2].scatter(cond_gu_HD,post_gu_HD, label=pearsonr(cond_gu_HD,post_gu_HD))
    #     axs[0,2].plot(np.arange(xm,xM), np.arange(xm,xM), c='r', linestyle='--')
    #     axs[0,2].legend(loc="upper right")

    #     axs[1,0].scatter(pre_gu_LD,cond_gu_LD, c='r', label=pearsonr(pre_gu_LD,cond_gu_LD))
    #     axs[1,0].plot(np.arange(xm,xM), np.arange(xm,xM), c='r', linestyle='--')
    #     axs[1,0].legend(loc="upper right")
    #     axs[1,1].scatter(pre_gu_LD,post_gu_LD, c='r', label=pearsonr(pre_gu_LD,post_gu_LD))
    #     axs[1,1].plot(np.arange(xm,xM), np.arange(xm,xM), c='r', linestyle='--')
    #     axs[1,1].legend(loc="upper right")
    #     axs[1,2].scatter(cond_gu_LD,post_gu_LD, c='r', label=pearsonr(cond_gu_LD,post_gu_LD))
    #     axs[1,2].plot(np.arange(xm,xM), np.arange(xm,xM), c='r', linestyle='--')
    #     axs[1,2].legend(loc="upper right")
    #     plt.suptitle(att)
    #     plt.show()


    # ###################### SORTED PLOT GINI ##############################

    # for att in atts:
    #     fig, axs = plt.subplots(1,2, sharey=True)

    #     x_HD = np.arange(len(pre_HD[att].values))
    #     pre_gu_HD =  np.where(pre_HD[att].values>0, pre_HD[att].values, 0)
    #     cond_gu_HD = np.where(cond_HD[att].values>0, cond_HD[att].values, 0)
    #     post_gu_HD = np.where(post_HD[att].values>0, post_HD[att].values, 0)

    #     x_LD = np.arange(len(pre_LD[att].values))
    #     pre_gu_LD = np.where(pre_LD[att].values>0, pre_LD[att].values, 0)
    #     cond_gu_LD = np.where(cond_LD[att].values>0, cond_LD[att].values, 0)
    #     post_gu_LD = np.where(post_LD[att].values>0, post_LD[att].values, 0)


    #     axs[0].plot(x_HD,sorted(pre_gu_HD), label="PRE")
    #     axs[0].plot(x_HD,sorted(cond_gu_HD), label="COND")
    #     axs[0].plot(x_HD,sorted(post_gu_HD), label="POST")
    #     axs[0].hlines(y=0.5, color='r', xmin=0, xmax=32, linestyle='-')

    #     axs[1].plot(x_LD,sorted(pre_gu_LD), label="PRE")
    #     axs[1].plot(x_LD,sorted(cond_gu_LD), label="COND")
    #     axs[1].plot(x_LD,sorted(post_gu_LD), label="POST")
    #     axs[1].hlines(y=0.5, color='r', xmin=0, xmax=32, linestyle='-')

    #     plt.suptitle(att)
    #     plt.legend()
    #     plt.show()


    # # ###################### HIST GINI ##############################

    # for att in atts:
    #     fig, axs = plt.subplots(2,3, sharey=True, sharex=True)

    #     x_HD = np.arange(len(pre_HD[att].values))
    #     pre_gu_HD =  np.where(pre_HD[att].values>0, pre_HD[att].values, 0)
    #     cond_gu_HD = np.where(cond_HD[att].values>0, cond_HD[att].values, 0)
    #     post_gu_HD = np.where(post_HD[att].values>0, post_HD[att].values, 0)

    #     x_LD = np.arange(len(pre_LD[att].values))
    #     pre_gu_LD = np.where(pre_LD[att].values>0, pre_LD[att].values, 0)
    #     cond_gu_LD = np.where(cond_LD[att].values>0, cond_LD[att].values, 0)
    #     post_gu_LD = np.where(post_LD[att].values>0, post_LD[att].values, 0)

    #     mu1, std1 = norm.fit(pre_gu_HD) 
    #     mu2, std2 = norm.fit(cond_gu_HD) 
    #     mu3, std3 = norm.fit(post_gu_HD) 
    #     mu4, std4 = norm.fit(pre_gu_LD) 
    #     mu5, std5 = norm.fit(cond_gu_LD) 
    #     mu6, std6 = norm.fit(post_gu_LD) 
    #     xmin, xmax = plt.xlim()
    #     x = np.linspace(xmin, xmax, 100)

    #     axs[0,0].hist((pre_gu_HD),bins=10, density=True, label="PRE")
    #     axs[0,0].plot(x, norm.pdf(x, mu1, std1), 'k')
    #     axs[0,1].hist((cond_gu_HD),bins=10, density=True, label="COND")
    #     axs[0,1].plot(x, norm.pdf(x, mu2, std2), 'k')
    #     axs[0,2].hist((post_gu_HD),bins=10, density=True, label="POST")
    #     axs[0,2].plot(x, norm.pdf(x, mu3, std3), 'k')

    #     axs[1,0].hist((pre_gu_LD),bins=10, density=True, color="r", label="PRE")
    #     axs[1,0].plot(x, norm.pdf(x, mu4, std4), 'k')
    #     axs[1,1].hist((cond_gu_LD),bins=10, density=True, color="r", label="COND")
    #     axs[1,1].plot(x, norm.pdf(x, mu5, std5), 'k')
    #     axs[1,2].hist((post_gu_LD),bins=10, density=True, color="r", label="POST")
    #     axs[1,2].plot(x, norm.pdf(x, mu6, std6), 'k')

    #     plt.suptitle(att)
    #     plt.legend()
    #     plt.show()



    #  # # ######################### VIOLIN PLOT ##############################

    # ticks = ['HD_PRE', 'LD_PRE', 'HD_COND', 'LD_COND', 'HD_POST', 'LD_POST']
    # colors = ['red', 'red', 'black', 'black', 'green', 'green']

    # fig, axs = plt.subplots(2,3, sharey=True, sharex=True)
    # axs = axs.reshape(-1)
    # for (att,ax) in zip(atts, axs):
    #     vs = []
    #     for phase in [df_logs_PRE, df_logs_COND, df_logs_POST]:
    #         for group in GROUPS:
    #             v = phase[phase.group == group][att].values
    #             v = np.where(v >= 0, v, np.nanmean(v))
    #             # v = np.where(v > 0, np.log(v), 0)
    #             vs.append(v)
    #     violin_parts = ax.violinplot(vs, showmeans=True)
    #     ax.set_xticks(np.arange(1,7))
    #     ax.set_xticklabels(ticks)
    #     ax.set_title(att)
    #     ax.tick_params(rotation=50)

    #     for c, pc in zip(colors, violin_parts['bodies']):
    #         pc.set_facecolor(c)

    # plt.show()


    # #  ################### PERC DIFF VIOLIN PLOT ###########################

    # d1_HD = []
    # d2_HD = []
    # for pid in df_logs_HD.username.unique():
    #     v1 = df_logs_HD[(df_logs_HD.username == pid) & (df_logs_HD.phase == "PRE")][atts].values[0]
    #     v2 = df_logs_HD[(df_logs_HD.username == pid) & (df_logs_HD.phase == "COND")][atts].values[0]
    #     v3 = df_logs_HD[(df_logs_HD.username == pid) & (df_logs_HD.phase == "POST")][atts].values[0]
    #     d1_HD.append(v2-v1)
    #     d2_HD.append(v3-v1)

    # d1_LD = []
    # d2_LD = []
    # for pid in df_logs_LD.username.unique():
    #     v1 = df_logs_LD[(df_logs_LD.username == pid) & (df_logs_LD.phase == "PRE")][atts].values[0]
    #     v2 = df_logs_LD[(df_logs_LD.username == pid) & (df_logs_LD.phase == "COND")][atts].values[0]
    #     v3 = df_logs_LD[(df_logs_LD.username == pid) & (df_logs_LD.phase == "POST")][atts].values[0]
    #     d1_LD.append(v2-v1)
    #     d2_LD.append(v3-v1)


    # v1 = [[x[y] for x in d1_HD] for y in np.arange(len(atts))]
    # v2 = [[x[y] for x in d1_LD] for y in np.arange(len(atts))]
    # v3 = [[x[y] for x in d2_HD] for y in np.arange(len(atts))]
    # v4 = [[x[y] for x in d2_LD] for y in np.arange(len(atts))]

    # vs = [v1,v2,v3,v4]

    # colors = ['red', 'red', 'black', 'black']

    # fig, axs = plt.subplots(3,3, sharey=True, sharex=True)
    # axs = axs.reshape(-1)
    # for c, ax in enumerate(axs):
    #     violin_parts = ax.violinplot([x[c] for x in vs], showmeans=True)
    #     ax.set_xticks(np.arange(1,5))
    #     ax.set_xticklabels(["HD_PRE-COND", "LD_PRE-COND", "HD_COND-POST", "LD_COND-POST"])
    #     ax.set_title(atts[c])
    #     ax.tick_params(rotation=50)

    #     for c, pc in zip(colors, violin_parts['bodies']):
    #         pc.set_facecolor(c)

    # plt.show()


    # # ################### SCATTER NOVEL PLOT ###########################

    # for att in atts:
    #     fig, axs = plt.subplots(1,2)

    #     cond_gu_HD = np.where(cond_diff_HD[att].values>0, np.log(cond_diff_HD[att].values), 0)
    #     post_gu_HD = np.where(post_diff_HD[att].values>0, np.log(post_diff_HD[att].values), 0)

    #     cond_gu_LD = np.where(cond_diff_LD[att].values>0, np.log(cond_diff_LD[att].values), 0)
    #     post_gu_LD = np.where(post_diff_LD[att].values>0, np.log(post_diff_LD[att].values), 0)

    #     xm = np.min([np.min(x) for x in [cond_gu_HD,post_gu_HD,cond_gu_LD,post_gu_LD]])
    #     xM = np.max([np.max(x) for x in [cond_gu_HD,post_gu_HD,cond_gu_LD,post_gu_LD]])+1


    #     axs[0].scatter(cond_gu_HD,post_gu_HD, label=pearsonr(cond_gu_HD,post_gu_HD))
    #     axs[0].plot(np.arange(xm,xM), np.arange(xm,xM), c='r', linestyle='--')
    #     axs[0].legend(loc="upper right")

    #     axs[1].scatter(cond_gu_LD,post_gu_LD, c='r', label=pearsonr(cond_gu_LD,post_gu_LD))
    #     axs[1].plot(np.arange(xm,xM), np.arange(xm,xM), c='r', linestyle='--')
    #     axs[1].legend(loc="upper right")
    #     plt.suptitle(att)
    #     plt.show()


    # # ################### SORTED NOVEL PLOT ###########################
    # for att in atts:
    #     fig, axs = plt.subplots(1,2, sharey=True)

    #     x_HD = np.arange(len(cond_diff_HD[att].values))
    #     cond_gu_HD = np.where(cond_diff_HD[att].values>0, np.log(cond_diff_HD[att].values), 0)
    #     post_gu_HD = np.where(post_diff_HD[att].values>0, np.log(post_diff_HD[att].values), 0)

    #     x_LD = np.arange(len(cond_diff_LD[att].values))
    #     cond_gu_LD = np.where(cond_diff_LD[att].values>0, np.log(cond_diff_LD[att].values), 0)
    #     post_gu_LD = np.where(post_diff_LD[att].values>0, np.log(post_diff_LD[att].values), 0)

    #     axs[0].plot(x_HD,sorted(cond_gu_HD), label="COND")
    #     axs[0].plot(x_HD,sorted(post_gu_HD), label="POST")
    #     axs[0].hlines(y=0.5, color='r', xmin=0, xmax=32, linestyle='-')

    #     axs[1].plot(x_LD,sorted(cond_gu_LD), label="COND")
    #     axs[1].plot(x_LD,sorted(post_gu_LD), label="POST")
    #     axs[1].hlines(y=0.5, color='r', xmin=0, xmax=32, linestyle='-')

    #     plt.suptitle(att)
    #     plt.legend()
    #     plt.show()


    # #################### NOVEL VIOLIN PLOT #########################

    # ticks = ['HD_COND', 'LD_COND', 'HD_POST', 'LD_POST']
    # colors = ['red', 'red', 'black', 'black']

    # fig, axs = plt.subplots(2,3, sharey=True, sharex=True)
    # axs = axs.reshape(-1)
    # for (att,ax) in zip(atts, axs):
    #     vs = []
    #     for phase in [df_diff_COND, df_diff_POST]:
    #         for group in GROUPS:
    #             v = phase[phase.group == group][att].values
    #             # v = np.where(v >= 0, v, np.nanmean(v))
    #             v = np.where(v > 0, np.log(v), 0)
    #             vs.append(v)
    #     violin_parts = ax.violinplot(vs, showmeans=True)
    #     ax.set_xticks(np.arange(1,5))
    #     ax.set_xticklabels(ticks)
    #     ax.set_title(att)
    #     ax.tick_params(rotation=50)

    #     for c, pc in zip(colors, violin_parts['bodies']):
    #         pc.set_facecolor(c)

    # plt.show()


    # #################### MEAN+STD PLOT #########################

    # fig, axs = plt.subplots(2,3, sharey=True, sharex=True)
    # axs = axs.reshape(-1)  
    # for n, (att,ax) in enumerate(zip(atts, axs)):
    #     x = np.arange(3)
    #     vs = []
    #     for username in cond_HD.username.unique():
    #         v = df_logs_HD[df_logs_HD.username==username][att]
    #         vs.append(np.where(v>0, np.log(v), 0))

    #     m1 = np.nanmean([x[0] for x in vs])
    #     s1 = np.nanstd([x[0] for x in vs])
    #     m2 = np.nanmean([x[1] for x in vs])
    #     s2 = np.nanstd([x[1] for x in vs])
    #     m3 = np.nanmean([x[2] for x in vs])
    #     s3 = np.nanstd([x[2] for x in vs])

    #     ax.plot(x, [m1,m2,m3], '-' , label='HD')
    #     ax.fill_between(x, [m1-s1,m2-s2,m3-s3] ,[m1+s1,m2+s2,m3+s3], alpha=0.2)

    #     vs = []
    #     for username in cond_LD.username.unique():
    #         v = df_logs_LD[df_logs_LD.username==username][att]
    #         vs.append(np.where(v>0, np.log(v), 0))

    #     m1 = np.nanmean([x[0] for x in vs])
    #     s1 = np.nanstd([x[0] for x in vs])
    #     m2 = np.nanmean([x[1] for x in vs])
    #     s2 = np.nanstd([x[1] for x in vs])
    #     m3 = np.nanmean([x[2] for x in vs])
    #     s3 = np.nanstd([x[2] for x in vs])

    #     ax.plot(x, [m1,m2,m3], '-', label='LD')
    #     ax.fill_between(x, [m1-s1,m2-s2,m3-s3] ,[m1+s1,m2+s2,m3+s3], alpha=0.2)
    #     ax.set_title(att)
    #     ax.set_xticks(np.arange(0,3))
    #     ax.set_xticklabels(["PRE", "COND", "POST"])

    # plt.legend()
    # plt.show()


     # #################### JOIN O SCORE #########################

    # df_join_att = import_data("scores")
    # df_join_att.d_score = - df_join_att.d_score
    # df_join_cntx = import_data("cntx")

    # Dict_O = df_join_att[df_join_att.att_round=='00'][['PROLIFIC_PID','o_score']].set_index('PROLIFIC_PID')['o_score'].to_dict()
    # DictLB_PID = {}
    # with open("../data/prescreening/lb_usernames.csv", 'r') as inf:
    #     _reader = csv.reader(inf)
    #     next(_reader)
    #     for row in _reader:
    #         DictLB_PID[row[0]] = row[1]
    # DictLB_O = {k:Dict_O[v] for k,v in DictLB_PID.items() if v in Dict_O}

    # user_open = [k for k in DictLB_O if DictLB_O[k] >= 3]
    # user_clos = [k for k in DictLB_O if DictLB_O[k] < 3]

    # df_logs_O = df_logs[df_logs.username.isin(user_open)]
    # df_logs_C = df_logs[df_logs.username.isin(user_clos)]

    # pre_O = df_logs_O[df_logs_O.phase == 'PRE']
    # cond_O = df_logs_O[df_logs_O.phase == 'COND']
    # post_O = df_logs_O[df_logs_O.phase == 'POST']
    # pre_C = df_logs_C[df_logs_C.phase == 'PRE']
    # cond_C = df_logs_C[df_logs_C.phase == 'COND']
    # post_C = df_logs_C[df_logs_C.phase == 'POST']