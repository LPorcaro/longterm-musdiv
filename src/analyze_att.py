#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import plotly.figure_factory as ff
import matplotlib.pylab as pl

from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr,spearmanr
from tabulate import tabulate

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

    df_join_att = import_data("scores")
    df_join_att.d_score = - df_join_att.d_score
    df_join_cntx = import_data("cntx")


    hist_data = [df_join_att[(df_join_att.group=='HD') & (df_join_att.att_round ==x)].d_score.tolist() for x in ROUNDS]
    fig = ff.create_distplot(hist_data, ROUNDS_LAB, bin_size=.2, histnorm='probability')
    fig.show()

    hist_data = [df_join_att[(df_join_att.group=='LD') & (df_join_att.att_round ==x)].d_score.tolist() for x in ROUNDS]
    fig = ff.create_distplot(hist_data, ROUNDS_LAB, bin_size=.2, histnorm='probability')
    fig.show()


    print (len(df_join_att))
    pids = df_join_att[df_join_att.group=='HD'].sort_values(by=["d_score"]).PROLIFIC_PID.unique()
    # Add Missing Values
    for c, att in enumerate(ROUNDS):
        if df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c])].empty:
            df_join_att.loc[len(df_join_att.index)] = [pid,  
                df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c-1])].d_score.item(),
                df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c-1])].o_score.item(), 
                'HD', 
                att]
    print(len(df_join_att))

    colors = pl.cm.bwr(np.linspace(0,1,len(pids)))
    fig, axs = plt.subplots(1, 4, sharey=False)
    for n, pid in enumerate(pids):
        y = df_join_att[df_join_att.PROLIFIC_PID == pid].d_score.tolist()
        x = np.arange(len(y))
        for i in range(len(y)-1):
            axs[i].plot(x, y, c=colors[n])
            axs[i].set_xlim([x[i],x[i+1]])

    plt.subplots_adjust(wspace=0)
    plt.show()


    print (len(df_join_att))
    pids = df_join_att[df_join_att.group=='LD'].sort_values(by=["d_score", "att_round"]).PROLIFIC_PID.unique()
    # Add Missing Values
    for c, att in enumerate(ROUNDS):
        if df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c])].empty:
            df_join_att.loc[len(df_join_att.index)] = [pid,  
                df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c-1])].d_score.item(),
                df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c-1])].o_score.item(), 
                'LD', 
                att]
    print (len(df_join_att))

    pids = df_join_att[df_join_att.group=='LD'].sort_values(by=["d_score", "att_round"]).PROLIFIC_PID.unique()

    colors = pl.cm.bwr(np.linspace(0,1,len(pids)))
    fig, axs = plt.subplots(1, 4, sharey=False)
    for n, pid in enumerate(pids):
        y = df_join_att[df_join_att.PROLIFIC_PID == pid].d_score.tolist()
        x = np.arange(len(y))
        for i in range(len(y)-1):
            axs[i].plot(x, y, c=colors[n])
            axs[i].set_xlim([x[i],x[i+1]])

    plt.subplots_adjust(wspace=0)
    plt.show()
