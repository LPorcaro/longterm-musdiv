#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import plotly.figure_factory as ff
import matplotlib.pylab as pl
import csv

from corrstats import independent_corr
from collections import Counter
from itertools import chain
from sklearn.preprocessing import normalize
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr,spearmanr
from tabulate import tabulate


fsize = 20

ATT_FOLDER = "../data/attitudes"
LS_FOLDER = "../data/ls"

CONTEXTS = ["Relaxing", "Commuting", "Partying", "Running","Shopping",
            "Sleeping", "Studying", "Working"]

CONTEXTS_a = ["Relaxing", "Sleeping", "Studying", "Working"]
CONTEXTS_b = ["Commuting", "Partying", "Running", "Shopping"]
TRACK_FEATS = ["Tempo", "Danceability", "Acousticness", "Instrumentalness"]
ARTIST_FEATS = ["Gender", "Skin", "Origin", "Age"]
ROUNDS =  ["00", "01", "02", "03", "04", "10"]
ROUNDS_LAB = ['Pre', 'Week 1', "Week 2", "Week 3", "Week 4", "Post"]
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


def plot_correlation(df):
    """
    """
    # Compute correlation matrices
    for att in ['d_score', 'o_score']:
        CorrMatrixs = []
        for group in GROUPS:
            df_join_att_g = df[df.group == group]
            CorrMatrix = np.zeros((len(ROUNDS), len(ROUNDS)))
            for c1, att_round1 in enumerate(ROUNDS):
                for c2, att_round2 in enumerate(ROUNDS):
                    if c1 > c2:
                        continue
                    elif c1 == c2:
                        s = 1
                    else:
                        m1 = df_join_att_g[df_join_att_g.att_round == att_round1][att].mean()
                        s1 = df_join_att_g[df_join_att_g.att_round == att_round1][att].std()
                        m2 = df_join_att_g[df_join_att_g.att_round == att_round2][att].mean()
                        s2 = df_join_att_g[df_join_att_g.att_round == att_round2][att].std()

                        s = 0
                        n = 0
                        for pid in df_join_att_g.PROLIFIC_PID.unique():
                            p1 = df_join_att_g[(df_join_att_g.att_round == att_round1) & (df_join_att_g.PROLIFIC_PID == pid)][att]
                            p2 = df_join_att_g[(df_join_att_g.att_round == att_round2) & (df_join_att_g.PROLIFIC_PID == pid)][att]

                            if len(p1) == 0 or len(p2) == 0:
                                pass
                            else:
                                p1 = p1.item()
                                p2 = p2.item()
                                s += ((p1-m1)/s1) * ((p2-m2)/s2)
                                n += 1
                        s /= (n - 1)
    
                    CorrMatrix[c1,c2] = CorrMatrix[c2, c1] = s


            print(group, att)
            if group == 'HD':
                print(tabulate(np.triu(CorrMatrix, k=1), headers=ROUNDS_LAB, tablefmt="github"))
            elif group == 'LD':
                print(tabulate(np.tril(CorrMatrix, k=-1), headers=ROUNDS_LAB, tablefmt="github"))
            CorrMatrixs.append(CorrMatrix)

        CorrDiffMatrix = np.zeros((len(ROUNDS), len(ROUNDS)))
        for r1,_ in enumerate(ROUNDS):
            for r2,_ in enumerate(ROUNDS):
                    if r1 == r2:
                        CorrDiffMatrix[r1,r2] = 1
                    elif r1 > r2:
                        continue
                    else:
                        z, p = independent_corr(CorrMatrixs[0][r1,r2], CorrMatrixs[1][r1,r2], 47, 51)
                        CorrDiffMatrix[r1,r2] = CorrDiffMatrix[r2, r1] = p
        print('Fisher Corr')
        print(tabulate(np.triu(CorrDiffMatrix, k=1), headers=ROUNDS_LAB, tablefmt="github"))


    # Plot D-score
    fig, axs = plt.subplots(len(ROUNDS),len(ROUNDS),sharey=True,sharex=True)
    for group, c in zip(GROUPS, ["r","g"]):
        df_join_att_g = df[df.group == group]
        for c1, att_round1 in enumerate(ROUNDS):
            for c2, att_round2 in enumerate(ROUNDS):
                if c1 > c2:
                    continue
                elif c1 == c2:
                    axs[c1,c2].text(-0.5,0, ROUNDS_LAB[c1], fontsize = fsize)
                else:
                    x, y  = [],[]
                    for pid in df_join_att_g.PROLIFIC_PID.unique():
                        x_score = df_join_att_g[(df_join_att_g.att_round == att_round1) & (df_join_att_g.PROLIFIC_PID == pid)].d_score
                        y_score = df_join_att_g[(df_join_att_g.att_round == att_round2) & (df_join_att_g.PROLIFIC_PID == pid)].d_score
                        if len(x_score) > 0 and len(y_score) > 0:
                            x.append(x_score.item())
                            y.append(y_score.item())
                
                    x = np.array(x)
                    y = np.array(y)
                    a, b = np.polyfit(x, y, 1)
                    if group == 'HD':
                        axs[c1,c2].plot(np.arange(-2,2), a*np.arange(-2,2)+b, c=c)  
                        axs[c1,c2].scatter(x,y,c=c)
                        axs[c1,c2].set_ylim(-1,1)
                        axs[c1,c2].set_xlim(-1,1)
                        axs[c1,c2].plot(np.arange(-2,2), np.arange(-2,2), c='k', linestyle='--')
                    elif group == 'LD':
                        axs[c2,c1].plot(np.arange(-2,2), a*np.arange(-2,2)+b, c=c)  
                        axs[c2,c1].scatter(x,y,c=c, label=group)
                        axs[c2,c1].set_ylim(-1,1)
                        axs[c2,c1].set_xlim(-1,1)
                        axs[c2,c1].plot(np.arange(-2,2), np.arange(-2,2), c='k', linestyle='--')


    plt.suptitle("d-score correlation", fontsize = fsize)
    plt.show()


    # Plot O-score
    fig, axs = plt.subplots(len(ROUNDS),len(ROUNDS),sharey=True,sharex=True)
    for group, c in zip(GROUPS, ["r","g"]):
        df_join_att_g = df[df.group == group]
        for c1, att_round1 in enumerate(ROUNDS):
            for c2, att_round2 in enumerate(ROUNDS):
                if c1 > c2:
                    continue
                elif c1 == c2:
                    axs[c1,c2].text(1.5,2.5, ROUNDS_LAB[c1], fontsize = fsize)
                else:
                    x, y  = [],[]
                    for pid in df_join_att_g.PROLIFIC_PID.unique():
                        x_score = df_join_att_g[(df_join_att_g.att_round == att_round1) & (df_join_att_g.PROLIFIC_PID == pid)].o_score
                        y_score = df_join_att_g[(df_join_att_g.att_round == att_round2) & (df_join_att_g.PROLIFIC_PID == pid)].o_score

                        if len(x_score) > 0 and len(y_score) > 0:
                            x.append(x_score.item())
                            y.append(y_score.item())

                    x = np.array(x)
                    y = np.array(y)
                    a, b = np.polyfit(x, y, 1)
                    if group == 'HD':
                        axs[c1,c2].plot(np.arange(-1,6), a*np.arange(-1,6)+b, c=c)    
                        axs[c1,c2].scatter(x,y,c=c)
                        axs[c1,c2].set_ylim(-0.5,5.5)
                        axs[c1,c2].set_xlim(-0.5,5.5)
                        axs[c1,c2].plot(np.arange(-1,6), np.arange(-1,6), c='k', linestyle='--')
                    elif group == 'LD':
                        axs[c2,c1].plot(np.arange(-1,6), a*np.arange(-1,6)+b, c=c)    
                        axs[c2,c1].scatter(x,y,c=c, label=group)
                        axs[c2,c1].set_ylim(-0.5,5.5)
                        axs[c2,c1].set_xlim(-0.5,5.5)
                        axs[c2,c1].plot(np.arange(-1,6), np.arange(-1,6), c='k', linestyle='--')

    plt.suptitle("o-score correlation", fontsize = fsize)
    plt.show()


def plot_average():
    """
    """
    ########## Average d-score / o-score over time
    d_score_HD_mean = [df_join_att_HD[df_join_att_HD.att_round==r].d_score.mean() for r in ROUNDS]
    d_score_HD_std = [df_join_att_HD[df_join_att_HD.att_round==r].d_score.std() for r in ROUNDS]
    d_score_LD_mean = [df_join_att_LD[df_join_att_LD.att_round==r].d_score.mean() for r in ROUNDS]
    d_score_LD_std  = [df_join_att_LD[df_join_att_LD.att_round==r].d_score.std() for r in ROUNDS]
    o_score_HD_mean = [df_join_att_HD[df_join_att_HD.att_round==r].o_score.mean() for r in ROUNDS]
    o_score_HD_std = [df_join_att_HD[df_join_att_HD.att_round==r].o_score.std() for r in ROUNDS]
    o_score_LD_mean = [df_join_att_LD[df_join_att_LD.att_round==r].o_score.mean() for r in ROUNDS]
    o_score_LD_std  = [df_join_att_LD[df_join_att_LD.att_round==r].o_score.std() for r in ROUNDS]


    fig, ax = plt.subplots(1,2,sharex=True)
    x = np.arange(len(ROUNDS))
    ax[0].plot(x, d_score_HD_mean, '-' , label='HD', c='r')
    ax[0].hlines(d_score_HD_mean[0],0,5, colors='r',linestyles='dotted')
    ax[0].fill_between(x, np.sum([d_score_HD_mean, [-i for i in d_score_HD_std]],0),np.sum([d_score_HD_mean,d_score_HD_std],0), alpha=0.2, color='r')
    ax[0].plot(x, d_score_LD_mean, '-' , label='LD', c='g')
    ax[0].hlines(d_score_LD_mean[0],0,5, colors='g',linestyles='dotted')
    ax[0].fill_between(x, np.sum([d_score_LD_mean, [-i for i in d_score_LD_std]],0) ,np.sum([d_score_LD_mean,d_score_LD_std],0), alpha=0.2, color='g')
    
    ax[0].set_xlim(0,5)
    ax[0].set_xticklabels(ROUNDS_LAB, fontsize = fsize)
    ax[0].set_xticks(np.arange(6))
    ax[0].set_ylabel('d-score', fontsize = fsize)
    ax[0].set_xlabel('', fontsize = fsize)
    ax[0].set_title('Average d-score over time', fontsize = fsize)
    ax[0].grid()


    ax[1].plot(x, o_score_HD_mean, '-' , label='HD', c='r')
    ax[1].hlines(o_score_HD_mean[0],0,5, colors='r',linestyles='dotted')
    ax[1].fill_between(x, np.sum([o_score_HD_mean, [-i for i in o_score_HD_std]],0),np.sum([o_score_HD_mean,o_score_HD_std],0), alpha=0.2, color='r')
    ax[1].plot(x, o_score_LD_mean, '-' , label='LD', c='g')
    ax[1].hlines(o_score_LD_mean[0],0,5, colors='g',linestyles='dotted')
    ax[1].fill_between(x, np.sum([o_score_LD_mean, [-i for i in o_score_LD_std]],0) ,np.sum([o_score_LD_mean,o_score_LD_std],0), alpha=0.2, color='g')
    
    ax[1].set_xlim(0,5)
    ax[1].set_xticklabels(ROUNDS_LAB, fontsize = fsize)
    ax[1].set_xticks(np.arange(6))
    ax[1].set_ylabel('o-score', fontsize = fsize)
    ax[1].set_xlabel('', fontsize = fsize)
    ax[1].set_title('Average o-score over time', fontsize = fsize)
    plt.legend(fontsize = fsize)
    plt.grid()
    plt.show()


def plot_dscore_distr():
    """
    """
    ########### D-score distribution
    hist_data = [df_join_att[(df_join_att.group=='HD') & (df_join_att.att_round ==x)].d_score.tolist() for x in ROUNDS]
    fig = ff.create_distplot(hist_data, ROUNDS_LAB, bin_size=.2, histnorm='probability')
    fig.show()
    hist_data = [df_join_att[(df_join_att.group=='LD') & (df_join_att.att_round ==x)].d_score.tolist() for x in ROUNDS]
    fig = ff.create_distplot(hist_data, ROUNDS_LAB, bin_size=.2, histnorm='probability')
    fig.show()


def plot_link_dscore():
    """
    """
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
    # Plot
    colors = pl.cm.viridis(np.linspace(0,1,len(pids)))
    fig, axs = plt.subplots(2, 5, sharey=True)
    for n, pid in enumerate(pids):
        y = df_join_att[df_join_att.PROLIFIC_PID == pid].d_score.tolist()
        x = np.arange(len(y))
        for i in range(len(y)-1):
            axs[0,i].plot(x, y, c=colors[n], marker='x',alpha=0.6)
            axs[0,i].set_xlim([x[i],x[i+1]])
            axs[0,i].set_xticklabels([])
            axs[0,i].set_ylim([-1.5,1.5])
    axs[0,2].set_title('HD d-score', fontsize = fsize)


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
    # Plot
    colors = pl.cm.viridis(np.linspace(0,1,len(pids)))
    # fig, axs = plt.subplots(1, 4, sharey=False)
    for n, pid in enumerate(pids):
        y = df_join_att[df_join_att.PROLIFIC_PID == pid].d_score.tolist()
        x = np.arange(len(y))
        for i in range(len(y)-1):
            axs[1,i].plot(x, y, c=colors[n], marker='x',alpha=0.6)
            axs[1,i].set_xlim([x[i],x[i+1]])
            axs[1,i].set_xticks([x[i],x[i+1]])
            axs[1,i].set_xticklabels([ROUNDS_LAB[i],ROUNDS_LAB[i+1]], fontsize = fsize)
            axs[1,i].set_ylim([-1.5,1.5])
    axs[1,2].set_title('LD d-score', fontsize = fsize)
    plt.subplots_adjust(wspace=0)
    plt.show()


def plot_slope():
    """
    """
    ###########  Plot Slope D-score
    fig, axs = plt.subplots(1,2, sharey=True)
    for n, (group, c) in enumerate(zip(GROUPS, ["r","g"])):
        baseline = []
        slopes = []
        N_s = []
        df_group = df_join_att[df_join_att.group == group]
        for m, pid in enumerate(df_group.PROLIFIC_PID.unique()):
            y = [df_group[(df_group.PROLIFIC_PID == pid) & (df_group.att_round == att_round)
                 ].d_score.values for att_round in ROUNDS]
            y = [el[0] for el in y if el.size > 0]
            x = range(len(y))
            N = len(x)
            if N <= 3:
                continue

            baseline.append(y[0])
            slope = pg.linear_regression(x, y)
            slopes.append(slope[slope.names == 'x1'].coef.item())
            N_s.append(N*N*5)

        # [axs.text(x,y,z) for x,y,z in zip(baseline, slopes, df_group.PROLIFIC_PID.unique().tolist())]
        axs[n].scatter(baseline, slopes, color=c, label=group, s=N_s, facecolors='none')

        mean = np.mean(slopes)
        std = np.std(slopes)
        x = np.linspace(-1,1,10)
        y = np.zeros(len(x))+mean
        axs[n].plot(x, y, c=c, linestyle='dotted')
        axs[n].fill_between(x,y-std, y+std, alpha=0.3, color=c)


        axs[n].set_xlabel("Baseline d-score (pre)", fontsize=fsize)
        axs[n].set_ylabel("Slope d-score", fontsize=fsize)
        axs[n].plot(range(-1,2), np.zeros(len(range(-1,2))), linestyle='--', c='k')
        axs[n].set_xlim(-1,1)
        axs[n].legend(fontsize=fsize)
    plt.show()



    ###########  Plot Slope 0-score
    fig, axs = plt.subplots(1,2, sharey=True)
    for n, (group, c) in enumerate(zip(GROUPS, ["r","g"])):
        baseline = []
        slopes = []
        N_s = []
        df_group = df_join_att[df_join_att.group == group]
        for m, pid in enumerate(df_group.PROLIFIC_PID.unique()):
            y = [df_group[(df_group.PROLIFIC_PID == pid) & (df_group.att_round == att_round)
                 ].o_score.values for att_round in ROUNDS]
            y = [el[0] for el in y if el.size > 0]
            x = range(len(y))
            N = len(x)
            if N <= 3:
                continue

            baseline.append(y[0])
            slope = pg.linear_regression(x, y)
            slopes.append(slope[slope.names == 'x1'].coef.item())
            N_s.append(N*N*5)

        # [axs.text(x,y,z) for x,y,z in zip(baseline, slopes, df_group.PROLIFIC_PID.unique().tolist())]
        axs[n].scatter(baseline, slopes, color=c, label=group, s=N_s, facecolors='none')

        mean = np.mean(slopes)
        std = np.std(slopes)
        x = np.linspace(-0.5, 5.5, 10)
        y = np.zeros(len(x))+mean
        axs[n].plot(x, y, c=c, linestyle='dotted')
        axs[n].fill_between(x,y-std, y+std, alpha=0.3, color=c)


        axs[n].set_xlabel("Baseline o-score (pre)", fontsize=fsize)
        axs[n].set_ylabel("Slope o-score", fontsize=fsize)
        axs[n].plot(range(-1, 7), np.zeros(len(range(-1,7))), linestyle='--', c='k')
        axs[n].set_xlim(-0.5,5.5)
        axs[n].legend(fontsize=fsize)
    plt.show()


def plot_oscore_distr():
    """
    """
    ########### O-score distribution
    pids = df_join_att[df_join_att.group=='HD'].sort_values(by=["d_score"]).PROLIFIC_PID.unique()
    x = np.arange(6)
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

    ax.bar(x, x0n, width, color = colors[0], label='HD')
    ax.bar(x, x1n, width, bottom=x0n, color = colors[7])
    ax.bar(x, x2n, width, bottom=[sum(x) for x in zip(x0n,x1n)], color = colors[14])
    ax.bar(x, x3n, width, bottom=[sum(x) for x in zip(x0n,x1n,x2n)], color = colors[21])
    ax.bar(x, x4n, width, bottom=[sum(x) for x in zip(x0n,x1n,x2n,x3n)], color = colors[28])
    ax.bar(x, x5n, width, bottom=[sum(x) for x in zip(x0n,x1n,x2n,x3n,x4n)], color = colors[35])

    pids = df_join_att[df_join_att.group=='LD'].sort_values(by=["d_score"]).PROLIFIC_PID.unique()
    colors = pl.cm.summer(np.linspace(0,1,len(pids)))
    counters = [Counter(df_join_att[(df_join_att.group=='LD') & (df_join_att.att_round ==x)].o_score.tolist()) for x in ROUNDS]
    x0ld = [x[0] for x in counters]
    x1ld = [x[1] for x in counters]
    x2ld = [x[2] for x in counters]
    x3ld = [x[3] for x in counters]
    x4ld = [x[4] for x in counters]
    x5ld = [x[5] for x in counters]

    x0ldn,x1ldn,x2ldn,x3ldn,x4ldn,x5ldn = scale_P(np.array([x0ld,x1ld,x2ld,x3ld,x4ld,x5ld]))

    ax.bar(x+width, x0ldn, width, color = colors[0], label='LD')
    ax.bar(x+width, x1ldn, width, bottom=x0ldn, color = colors[7])
    ax.bar(x+width, x2ldn, width, bottom=[sum(x) for x in zip(x0ldn,x1ldn)], color = colors[14])
    ax.bar(x+width, x3ldn, width, bottom=[sum(x) for x in zip(x0ldn,x1ldn,x2ldn)], color = colors[21])
    ax.bar(x+width, x4ldn, width, bottom=[sum(x) for x in zip(x0ldn,x1ldn,x2ldn,x3ldn)], color = colors[28])
    ax.bar(x+width, x5ldn, width, bottom=[sum(x) for x in zip(x0ldn,x1ldn,x2ldn,x3ldn,x4ldn)], color = colors[35])
    ax.set_xticks(x+width/2)
    ax.set_xticklabels(ROUNDS_LAB, fontsize=fsize)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)


    # a = [x0n,x1n,x2n,x3n,x4n,x5n]
    # b = [x0ld,x1ld,x2ld,x3ld,x4ld,x5ld]
    # a = ([np.sum([x0n,x1n],0), np.sum([x2n,x3n],0), np.sum([x4n,x5n],0)])
    # b = ([np.sum([x0ldn,x1ldn],0), np.sum([x2ldn,x3ldn],0), np.sum([x4ldn,x5ldn],0)])
    a = ([np.sum([x0n,x1n,x2n],0), np.sum([x3n,x4n,x5n],0)])
    b = ([np.sum([x0ldn,x1ldn,x2ldn],0), np.sum([x3ldn, x4ldn,x5ldn],0)])

    celltext = []
    for c in range(2):
        celltext.append(list(map(float, chain.from_iterable([[round(a[c][i],1)]+[round(b[c][i],1)] for i in range(6)]))))

    t = ax.table(cellText=celltext,
                 rowLabels=["0-2", "3-5"],
                 cellLoc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(18)

    ax.set_title('o-score distribution',fontsize=fsize)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.ylim([0,100])
    plt.legend(fontsize=fsize)
    plt.show()


def plot_score_corr():
    """
    """ 
    ############# O-score vs d-score
    fig, ax = plt.subplots(1,2,sharey=True)

    rho1,p1 = pearsonr(df_join_att_HD.o_score, df_join_att_HD.d_score)
    rho2,p2 = pearsonr(df_join_att_LD.o_score, df_join_att_LD.d_score)

    ax[0].scatter(df_join_att_HD.o_score, df_join_att_HD.d_score, c='r', facecolors='none',label=r"HD, $\rho ={:.2f}$".format(rho1))
    v = ax[0].violinplot([df_join_att_HD[df_join_att_HD.o_score == x].d_score for x in range(6)], positions=range(6), showmeans=True)
    for pc in v['bodies']:
        pc.set_color('r')
        pc.set_facecolor('r')
        pc.set_edgecolor('r')

    a, b = np.polyfit(df_join_att_HD.o_score, df_join_att_HD.d_score, 1)
    ax[0].plot(np.arange(-1,7), a*np.arange(-1,7)+b,color='r', linestyle='--')  
    ax[0].set_xlabel('o-score', fontsize = fsize)
    ax[0].set_ylabel('d-score', fontsize = fsize)
    ax[0].grid()
    ax[0].set_xlim(-0.5,5.5)
    ax[0].legend(fontsize = fsize)

    ax[1].scatter(df_join_att_LD.o_score, df_join_att_LD.d_score, c='g', facecolors='none',label=r"LD, $\rho ={:.2f}$".format(rho2))
    v = ax[1].violinplot([df_join_att_LD[df_join_att_LD.o_score == x].d_score for x in range(6)], positions=range(6), showmeans=True)
    for pc in v['bodies']:
        pc.set_color('g')
        pc.set_facecolor('g')
        pc.set_edgecolor('g')

    a, b = np.polyfit(df_join_att_LD.o_score, df_join_att_LD.d_score, 1)
    ax[1].plot(np.arange(-1,7), a*np.arange(-1,7)+b,color='g', linestyle='--')  
    ax[1].set_xlabel('o-score', fontsize = fsize)
    ax[1].grid()
    ax[1].set_xlim(-0.5,5.5)
    ax[1].legend(fontsize = fsize)

    plt.suptitle('d-score VS o-score', fontsize = fsize)
    plt.show()

   

if __name__ == "__main__":

    df_join_att = import_data("scores")
    df_join_att.d_score = - df_join_att.d_score
    df_join_cntx = import_data("cntx")
    df_join_att = filter_dataframe(df_join_att, 'inc', 'pid')



    df_join_att_HD = df_join_att[df_join_att.group == 'HD']
    df_join_att_LD = df_join_att[df_join_att.group == 'LD']

    # plot_correlation(df_join_att)
    # plot_average()
    # plot_dscore_distr()
    # plot_link_dscore()
    # plot_slope()
    # plot_oscore_distr()
    # plot_score_corr()








