#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import plotly.figure_factory as ff
import matplotlib.pylab as pl

from collections import Counter
from itertools import chain
from sklearn.preprocessing import normalize
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


def scale_P(A):
    return(A*100/sum(A))


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


    df_join_att_HD = df_join_att[df_join_att.group == 'HD']
    df_join_att_LD = df_join_att[df_join_att.group == 'LD']


    ########### D-score distribution
    hist_data = [df_join_att[(df_join_att.group=='HD') & (df_join_att.att_round ==x)].d_score.tolist() for x in ROUNDS]
    fig = ff.create_distplot(hist_data, ROUNDS_LAB, bin_size=.2, histnorm='probability')
    fig.show()
    hist_data = [df_join_att[(df_join_att.group=='LD') & (df_join_att.att_round ==x)].d_score.tolist() for x in ROUNDS]
    fig = ff.create_distplot(hist_data, ROUNDS_LAB, bin_size=.2, histnorm='probability')
    fig.show()


    ########### Link D-score
    print (len(df_join_att))
    pids = df_join_att[df_join_att.group=='HD'].sort_values(by=["d_score"]).PROLIFIC_PID.unique()
    # Add Missing Values
    for pid in pids:
        for c, att in enumerate(ROUNDS):
            if df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c])].empty:
                df_join_att.loc[len(df_join_att.index)] = [pid,  
                    df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c-1])].d_score.item(),
                    df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c-1])].o_score.item(), 
                    'HD', 
                    att]
    print(len(df_join_att))
    # Plot
    colors = pl.cm.summer(np.linspace(0,1,len(pids)))
    fig, axs = plt.subplots(2, 4, sharey=False)
    for n, pid in enumerate(pids):
        y = df_join_att[df_join_att.PROLIFIC_PID == pid].d_score.tolist()
        x = np.arange(len(y))
        for i in range(len(y)-1):
            axs[0,i].plot(x, y, c=colors[n])
            axs[0,i].set_xlim([x[i],x[i+1]])
            axs[0,i].set_xticklabels([])
            axs[0,i].set_ylim([-1.5,1.5])
    axs[0,1].set_title('High-Diversity - D-score')


    # Add Missing Values
    print (len(df_join_att))
    pids = df_join_att[df_join_att.group=='LD'].sort_values(by=["d_score"]).PROLIFIC_PID.unique()
    for pid in pids:
        for c, att in enumerate(ROUNDS):
            if df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c])].empty:
                df_join_att.loc[len(df_join_att.index)] = [pid,  
                    df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c-1])].d_score.item(),
                    df_join_att[(df_join_att.PROLIFIC_PID == pid) & (df_join_att.att_round == ROUNDS[c-1])].o_score.item(), 
                    'LD', 
                    att]
    print (len(df_join_att))
    # Plot
    colors = pl.cm.summer(np.linspace(0,1,len(pids)))
    # fig, axs = plt.subplots(1, 4, sharey=False)
    for n, pid in enumerate(pids):
        y = df_join_att[df_join_att.PROLIFIC_PID == pid].d_score.tolist()
        x = np.arange(len(y))
        for i in range(len(y)-1):
            axs[1,i].plot(x, y, c=colors[n])
            axs[1,i].set_xlim([x[i],x[i+1]])
            axs[1,i].set_xticks([x[i],x[i+1]])
            axs[1,i].set_xticklabels([ROUNDS_LAB[i],ROUNDS_LAB[i+1]])
            axs[1,i].set_ylim([-1.5,1.5])
    axs[1,1].set_title('Low-Diversity - D-score')
    plt.subplots_adjust(wspace=0)
    plt.show()


    ###########  Plot Slope D-score
    fig, axs = plt.subplots(1,2, sharey=True)
    for n, (group, c) in enumerate(zip(GROUPS, ["b","g"])):
        baseline = []
        slopes = []
        df_group = df_join_att[df_join_att.group == group]
        for m, pid in enumerate(df_group.PROLIFIC_PID.unique()):
            y = [df_group[(df_group.PROLIFIC_PID == pid) & (df_group.att_round == att_round)
                 ].d_score.values for att_round in ROUNDS]
            y = [el[0] for el in y if el.size > 0]
            x = range(len(y))
            if len(x) <= 3:
                continue

            baseline.append(y[0])
            slope = pg.linear_regression(x, y)
            slopes.append(slope[slope.names == 'x1'].coef.item())

        # [axs.text(x,y,z) for x,y,z in zip(baseline, slopes, df_group.PROLIFIC_PID.unique().tolist())]
        axs[n].scatter(baseline, slopes, color=c, label=group)

        # a, b = np.polyfit(baseline, slopes, 1)
        # axs[n].plot(np.arange(-1,2), a*np.arange(-1,2)+b, c=c)  

        mean = np.mean(slopes)
        std = np.std(slopes)
        x = np.linspace(-1,1,10)
        y = np.zeros(len(x))+mean
        axs[n].plot(x, y, c=c, linestyle='dotted')
        axs[n].fill_between(x,y-std, y+std, alpha=0.4, color=c)


        axs[n].set_xlabel("Baseline D-score (pre)")
        axs[n].set_ylabel("Slope D-score")
        axs[n].plot(range(-1,2), np.zeros(len(range(-1,2))), linestyle='--', c='r')
        axs[n].set_xlim(-1,1)
        axs[n].legend()
    plt.show()



    ########### O-score distribution
    fig, ax = plt.subplots()
    colors = pl.cm.autumn(np.linspace(0,1,len(pids)))
    counters = [Counter(df_join_att[(df_join_att.group=='HD') & (df_join_att.att_round ==x)].o_score.tolist()) for x in ROUNDS]
    width = 0.35
    x0hd = [x[0] for x in counters]
    x1hd = [x[1] for x in counters]
    x2hd = [x[2] for x in counters]
    x3hd = [x[3] for x in counters]
    x4hd = [x[4] for x in counters]
    x5hd = [x[5] for x in counters]

    x0n,x1n,x2n,x3n,x4n,x5n = scale_P(np.array([x0hd,x1hd,x2hd,x3hd,x4hd,x5hd]))

    ax.bar(np.arange(5), x0n, width, color = colors[0], label='HD')
    ax.bar(np.arange(5), x1n, width, bottom=x0n, color = colors[10])
    ax.bar(np.arange(5), x2n, width, bottom=[sum(x) for x in zip(x0n,x1n)], color = colors[20])
    ax.bar(np.arange(5), x3n, width, bottom=[sum(x) for x in zip(x0n,x1n,x2n)], color = colors[30])
    ax.bar(np.arange(5), x4n, width, bottom=[sum(x) for x in zip(x0n,x1n,x2n,x3n)], color = colors[40])
    ax.bar(np.arange(5), x5n, width, bottom=[sum(x) for x in zip(x0n,x1n,x2n,x3n,x4n)], color = colors[50])

    colors = pl.cm.winter(np.linspace(0,1,len(pids)))
    counters = [Counter(df_join_att[(df_join_att.group=='LD') & (df_join_att.att_round ==x)].o_score.tolist()) for x in ROUNDS]
    x0 = [x[0] for x in counters]
    x1 = [x[1] for x in counters]
    x2 = [x[2] for x in counters]
    x3 = [x[3] for x in counters]
    x4 = [x[4] for x in counters]
    x5 = [x[5] for x in counters]

    x0n,x1n,x2n,x3n,x4n,x5n = scale_P(np.array([x0,x1,x2,x3,x4,x5]))

    ax.bar(np.arange(5)+width, x0n, width, color = colors[0], label='LD')
    ax.bar(np.arange(5)+width, x1n, width, bottom=x0n, color = colors[10])
    ax.bar(np.arange(5)+width, x2n, width, bottom=[sum(x) for x in zip(x0n,x1n)], color = colors[20])
    ax.bar(np.arange(5)+width, x3n, width, bottom=[sum(x) for x in zip(x0n,x1n,x2n)], color = colors[30])
    ax.bar(np.arange(5)+width, x4n, width, bottom=[sum(x) for x in zip(x0n,x1n,x2n,x3n)], color = colors[40])
    ax.bar(np.arange(5)+width, x5n, width, bottom=[sum(x) for x in zip(x0n,x1n,x2n,x3n,x4n)], color = colors[50])
    ax.set_xticks(np.arange(5)+width/2)
    ax.set_xticklabels(ROUNDS_LAB)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)


    a = [x0hd,x1hd,x2hd,x3hd,x4hd,x5hd]
    b = [x0,x1,x2,x3,x4,x5]
    celltext = []
    for c in range(6):
        celltext.append(list(map(int, chain.from_iterable([[a[c][i]]+[b[c][i]] for i in range(5)]))))

    ax.table(cellText=celltext,
             rowLabels=np.arange(6),
             cellLoc='center'
             )

    ax.set_title('O-score Distribution')
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.ylim([0,100])
    plt.legend()
    plt.show()







    fig = px.density_heatmap(df_join_att_HD, x="d_score", y="o_score", marginal_x="box", marginal_y="box", range_x =[-1,1],range_y =[0,5],nbinsx=10)
    fig.show()
    fig = px.density_heatmap(df_join_att_LD, x="d_score", y="o_score", marginal_x="box", marginal_y="box", range_x =[-1,1],range_y =[0,5],nbinsx=10)
    fig.show()

    fig = px.density_heatmap(df_join_att_HD, x="d_score", y="o_score", marginal_x="histogram", marginal_y="histogram",range_x =[-1.5,1.5], range_y =[-0.5,5.5], title='HD')
    fig.show()
    fig = px.density_heatmap(df_join_att_LD, x="d_score", y="o_score", marginal_x="histogram", marginal_y="histogram", range_x =[-1.5,1.5], range_y =[-0.5,5.5], title='LD')
    fig.show()
