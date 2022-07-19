#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import plot_likert as pl

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


def plot_scores(df):
    """
    """
    # Plot D-score Individual
    # fig, axs = plt.subplots(2, 4,sharey=True,sharex=True)
    # for n, (group, c) in enumerate(zip(GROUPS, ["b","g"])):
    #     df_group = df[df.group == group]
    #     for m, pid in enumerate(df_group.PROLIFIC_PID.unique()):
    #         y = [df_group[(df_group.PROLIFIC_PID == pid) & (df_group.att_round == att_round)
    #              ].d_score.values for att_round in ROUNDS]
    #         y = [el[0] for el in y if el.size > 0]
    #         x = range(len(y))
    #         axs[n,m].plot(x, y, c=c, label=group)
    #         axs[n,m].set_xticks(x)
    #         axs[n,m].set_xticklabels(ROUNDS_LAB[:len(x)])
    #         axs[n,m].tick_params(rotation=50)
    #         axs[n,m].set_title(pid)
    #         axs[n,m].grid()
    # axs[0,0].set_ylabel('D-score')
    # axs[1,0].set_ylabel('D-score')
    # axs[0,0].legend()
    # axs[1,0].legend() 
    # plt.ylim([-1,1])
    # plt.show()
    
    # Plot D-score joint
    for group, c in zip(GROUPS, ["b","g"]):
        df_group = df[df.group == group]
        for pid in df_group.PROLIFIC_PID.unique():
            y = [df_group[(df_group.PROLIFIC_PID == pid) & (df_group.att_round == att_round)
                 ].d_score.values for att_round in ROUNDS]
            y = [el[0] for el in y if el.size > 0]
            x = range(len(y))
            plt.plot(x, y, c=c, label=group)
    plt.title('D-score')
    plt.xticks(x, ROUNDS_LAB, horizontalalignment='right')
    plt.legend()
    plt.grid()
    plt.ylim([-1,1])
    plt.show()
    # Plot Slope D-score
    fig, axs = plt.subplots()
    for n, (group, c) in enumerate(zip(GROUPS, ["b","g"])):
        baseline = []
        slopes = []
        df_group = df[df.group == group]
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
        axs.scatter(baseline, slopes, color=c, label=group)

    axs.set_xlabel("Baseline D-score (pre)")
    axs.set_ylabel("Slope D-score")
    axs.plot(range(-1,2), np.zeros(len(range(-1,2))), linestyle='--', c='r')
    axs.set_xlim(-1,1)
    plt.legend()
    plt.show()

    # # Plot O-score Individual
    # fig, axs = plt.subplots(2,4,sharey=True,sharex=True)
    # for n, (group, c) in enumerate(zip(GROUPS, ["b","g"])):
    #     df_group = df[df.group == group]
    #     for m, pid in enumerate(df_group.PROLIFIC_PID.unique()):
    #         y = [df_group[(df_group.PROLIFIC_PID == pid) & (df_group.att_round == att_round)
    #              ].o_score.values for att_round in ROUNDS]
    #         y = [el[0] for el in y if el.size > 0]
    #         x = range(len(y))
    #         axs[n,m].plot(x, y, c=c, label=group)
    #         axs[n,m].set_xticks(x)
    #         axs[n,m].set_xticklabels(ROUNDS_LAB[:len(x)])
    #         axs[n,m].tick_params(rotation=50)
    #         axs[n,m].set_title(pid)
    #         axs[n,m].grid()
    # axs[0,0].set_ylabel('D-score')
    # axs[1,0].set_ylabel('D-score')
    # axs[0,0].legend()
    # axs[1,0].legend() 
    # plt.show()
    # Plot O-score join
    for group, c in zip(GROUPS, ["b","g"]):
        df_group = df[df.group == group]
        for pid in df_group.PROLIFIC_PID.unique():
            y = [df_group[(df_group.PROLIFIC_PID == pid) & (df_group.att_round == att_round)
                 ].o_score.values for att_round in ROUNDS]
            y = [el[0] for el in y if el.size > 0]
            x = range(len(y))
            plt.plot(x, y, c=c, label=group)
    plt.title('O-score')
    plt.xticks(x, ROUNDS_LAB, horizontalalignment='right')
    plt.legend()
    plt.grid()
    plt.show()
    # Plot Slope O-score
    fig, axs = plt.subplots()
    for n, (group, c) in enumerate(zip(GROUPS, ["b","g"])):
        baseline = []
        slopes = []
        df_group = df[df.group == group]
        for m, pid in enumerate(df_group.PROLIFIC_PID.unique()):
            y = [df_group[(df_group.PROLIFIC_PID == pid) & (df_group.att_round == att_round)
                 ].o_score.values for att_round in ROUNDS]
            y = [el[0] for el in y if el.size > 0]
            x = range(len(y))
            if len(x) <= 3:
                continue
            baseline.append(y[0])
            slope = pg.linear_regression(x, y)
            slopes.append(slope[slope.names == 'x1'].coef.item())

        [axs.text(x,y,z) for x,y,z in zip(baseline, slopes, df_group.PROLIFIC_PID.unique().tolist())]
        axs.scatter(baseline, slopes, color=c, label=group)

    axs.set_xlabel("Baseline O-score (pre)")
    axs.set_ylabel("Slope O-score")
    axs.plot(range(-1,6), np.zeros(len(range(-1,6))), linestyle='--', c='r')
    axs.set_xlim(-0.5,5.5)
    plt.legend()
    plt.show()

    # Plot o-score VS d-score
    fig, axs = plt.subplots(1,len(ROUNDS),sharey=True)
    for n, att_round in enumerate(ROUNDS):
        df_round = df[df.att_round == att_round]
        for group, c in zip(GROUPS, ["b","g"]):
            x = df_round[df_round.group == group].o_score
            y = df_round[df_round.group == group].d_score
            axs[n].scatter(x, y, label=group, s=50, c=c, marker='x')
        axs[n].legend(loc=2)
        axs[n].hlines(y=0, color='r', xmin=-2, xmax=6, linestyle='-')
        axs[n].vlines(x=2.5, color='r', ymin=-1, ymax=1, linestyle='-')
        axs[n].set_ylim(-1,1)
        axs[n].set_xlim(-0.5,5.5)
    axs[1].set_xlabel("O-score")
    axs[0].set_ylabel("D-score")
    axs[0].set_title(label="Pre")
    axs[1].set_title(label="Week 1")
    axs[2].set_title(label="Week 2")
    axs[3].set_title(label="Week 3")
    axs[4].set_title(label="Week 4")
    # axs[5].set_title(label="Post")
    plt.show()


def plot_cntx(df):
    """
    """
    colors = ['r','b','b','b','b', 'r', 'y', 'g','g', 'g', 'g','y']
    # Plot Context Boxplots
    fig, axs = plt.subplots(2,4, sharex=True, sharey=True)
    axs = axs.reshape(-1)
    for column, ax in zip(CONTEXTS, axs):
        bp_dict = df.boxplot(by=['group','att_round'],
                                       column=column,
                                       layout=(2,2),
                                       return_type='both',
                                       patch_artist = True,
                                       ax=ax)
        ax.set_yticklabels(['Disagree', '', '', '', 'Agree'])
        ax.set_yticks([1,2,3,4,5])
        ax.set_xlabel('')
        ax.tick_params(rotation=50)
        for row_key, (ax,row) in bp_dict.iteritems():
            for i, box in enumerate(row['boxes']):
                box.set_facecolor(colors[i])
        plt.suptitle("In which contexts would you listen to Electronic Music?")
    plt.show()
    # # Plot Context Likert
    # fig, axs = plt.subplots(2, len(ROUNDS), sharey=True,  sharex=True)
    # for x1, (group, col) in enumerate(zip(GROUPS, [pl.colors.likert5, pl.colors.default_with_darker_neutral])):
    #     for x2, (att_round, att_round_l) in enumerate(zip(ROUNDS, ROUNDS_LAB)):
    #         df_cut = df[(df.att_round == att_round) & (df.group == group)][CONTEXTS]
    #         pl.plot_likert(df_cut, [1,2,3,4,5], 
    #                         plot_percentage=True,
    #                         bar_labels=True, 
    #                         bar_labels_color="snow", 
    #                         colors=col, 
    #                         ax=axs[x1,x2])

    #         axs[x1,x2].get_legend().remove()
    #         axs[x1,x2].set_xlabel(att_round_l)

    # plt.suptitle("In which contexts would you listen to Electronic Music?")
    # plt.show()


    # Plot Tracks Boxplots
    fig, axs = plt.subplots(2,2, sharex=True)
    axs = axs.reshape(-1)
    for column, ax in zip(TRACK_FEATS, axs):
        bp_dict = df.boxplot(by=['group','att_round'],
                                       column=column,
                                       layout=(2,2),
                                       return_type='both',
                                       patch_artist = True,
                                       ax=ax)
        if column == 'Tempo':
            ax.set_yticklabels(['Slow', '', '', '', 'Fast'])
        else:
            ax.set_yticklabels(['Low', '', '', '', 'High'])

        ax.set_xlabel('')
        ax.set_yticks([1,2,3,4,5])
        for row_key, (ax,row) in bp_dict.iteritems():
            for i, box in enumerate(row['boxes']):
                box.set_facecolor(colors[i])
        plt.suptitle("Which features do you associate Electronic Music?")
    plt.show()
    # # Plot Tracks Likert
    # fig, axs = plt.subplots(2,len(ROUNDS), sharey=True,  sharex=True)
    # for x1, (group, col) in enumerate(zip(GROUPS, [pl.colors.likert5, pl.colors.default_with_darker_neutral])):
    #     for x2, (att_round, att_round_l) in enumerate(zip(ROUNDS, ROUNDS_LAB)):
    #         df_cut = df[(df.att_round == att_round) & (df.group == group)][TRACK_FEATS]
    #         pl.plot_likert(df_cut, [1,2,3,4,5], 
    #                         plot_percentage=True,
    #                         bar_labels=True, 
    #                         bar_labels_color="snow", 
    #                         colors=col, 
    #                         ax=axs[x1,x2])

    #         axs[x1,x2].get_legend().remove()
    #         axs[x1,x2].set_xlabel(att_round_l)

    # plt.suptitle("Which features do you associate Electronic Music?")
    # plt.show()


    # Plot Artists Boxplots
    fig, axs = plt.subplots(2,2, sharex=True)
    axs = axs.reshape(-1)
    for column, ax in zip(ARTIST_FEATS, axs):
        bp_dict = df.boxplot(by=['group','att_round'],
                                       column=column,
                                       layout=(2,2),
                                       return_type='both',
                                       patch_artist = True,
                                       ax=ax)
        if column == 'Gender':
            ax.set_yticklabels(['Female', '', '', '', 'Male'])
        elif column == 'Skin':
            ax.set_yticklabels(['White', '', '', '', 'Black'])
        elif column == 'Origin':
            ax.set_yticklabels(['Low-income', '', '', '', 'High-income'])
        elif column == 'Age':
            ax.set_yticklabels(['<40', '', '', '', '>40'])
        ax.set_xlabel('')
        ax.set_yticks([1,2,3,4,5])
        for row_key, (ax,row) in bp_dict.iteritems():
            for i, box in enumerate(row['boxes']):
                box.set_facecolor(colors[i])
        plt.suptitle("Which characteristics do you associate Electronic Music artists?")
    plt.show()
    # # Plot Artists Likert
    # fig, axs = plt.subplots(2,len(ROUNDS), sharey=True, sharex=True)
    # for x1, (group, col) in enumerate(zip(GROUPS, [pl.colors.likert5, pl.colors.default_with_darker_neutral])):
    #     for x2, (att_round, att_round_l) in enumerate(zip(ROUNDS, ROUNDS_LAB)):
    #         df_cut = df[(df.att_round == att_round) & (df.group == group)][ARTIST_FEATS]
    #         pl.plot_likert(df_cut, [1,2,3,4,5], 
    #                         plot_percentage=True,
    #                         bar_labels=True, 
    #                         bar_labels_color="snow", 
    #                         colors=col, 
    #                         ax=axs[x1,x2])

    #         axs[x1,x2].get_legend().remove()
    #         axs[x1,x2].set_xlabel(att_round_l)

    # plt.suptitle("Which characteristics do you associate Electronic Music artists?")
    # plt.show()

    from corrstats import independent_corr

    # INTRA-RATER CORRELATION
    for feat in [CONTEXTS_a, CONTEXTS_b, TRACK_FEATS, ARTIST_FEATS]:
        print()
        print(feat)
        CorrMatrixs = []
        for n, (group, c) in enumerate(zip(GROUPS, ["b","g"])):
            print(group)
            CorrMatrix = np.zeros((len(ROUNDS), len(ROUNDS)))
            for r1,_ in enumerate(ROUNDS):
                for r2,_ in enumerate(ROUNDS):
                    if r1 == r2:
                        rho_mean = 1
                    elif r1 > r2:
                        continue
                    else:
                        rhos = []
                        df_group = df[df.group == group]
                        df_join_pid = df_group.groupby(by='PROLIFIC_PID')

                        for m, (pid, df_pid) in enumerate(df_join_pid):
                            a = df_pid[df_pid.att_round==ROUNDS[r1]][feat].values
                            b = df_pid[df_pid.att_round==ROUNDS[r2]][feat].values

                            if len(a) == 0 or len(b) == 0: # MISSING DATA
                                continue
                            else:
                                a = a[0]
                                b = b[0]

                                if (a == b).all():
                                    rho = 1
                                else:
                                    rho,p = pearsonr(a,b)
                                
                            if np.isnan(rho):
                                rho = 0.01

                            if rho == 0:
                                rho = 0.01
                            elif rho == 1:
                                rho = 0.99
                            elif rho == -1:
                                rho = -0.99

                            rhos.append(np.arctanh(rho))  

                        rho_mean = np.average(rhos)

                    CorrMatrix[r1,r2] = CorrMatrix[r2, r1] = np.tanh(rho_mean)
            CorrMatrixs.append(CorrMatrix)
            print(tabulate(np.triu(CorrMatrix, k=1), headers=ROUNDS_LAB, tablefmt="github"))
        # permutation_test(pd.DataFrame(CorrMatrixs[0]), pd.DataFrame(CorrMatrixs[1]))

        CorrDiffMatrix = np.zeros((len(ROUNDS), len(ROUNDS)))
        for r1,_ in enumerate(ROUNDS):
            for r2,_ in enumerate(ROUNDS):
                    if r1 == r2:
                        CorrDiffMatrix[r1,r2] = 1
                    elif r1 > r2:
                        continue
                    else:
                        z, p = independent_corr(CorrMatrixs[0][r1,r2], CorrMatrixs[1][r1,r2], 47,51)
                        CorrDiffMatrix[r1,r2] = CorrDiffMatrix[r2, r1] = p
        print('Fisher Corr')
        print(tabulate(np.triu(CorrDiffMatrix, k=1), headers=ROUNDS_LAB, tablefmt="github"))


def upper(df):
    '''Returns the upper triangle of a correlation matrix.
    You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.
    Args:
      df: pandas or numpy correlation matrix
    Returns:
      list of values from upper triangle
    '''
    try:
        assert(type(df)==np.ndarray)
    except:
        if type(df)==pd.DataFrame:
            df = df.values
        else:
            raise TypeError('Must be np.ndarray or pd.DataFrame')
    mask = np.triu_indices(df.shape[0], k=1)
    return df[mask]


def permutation_test(m1, m2):
    """Nonparametric permutation testing Monte Carlo"""
    np.random.seed(0)
    rhos = []
    n_iter = 5000
    true_rho, _ = spearmanr(upper(m1), upper(m2))
    # matrix permutation, shuffle the groups
    m_ids = list(m1.columns)
    m2_v = upper(m2)
    for iter in range(n_iter):
      np.random.shuffle(m_ids) # shuffle list 
      r, _ = spearmanr(upper(m1.loc[m_ids, m_ids]), m2_v)  
      rhos.append(r)
    perm_p = ((np.sum(np.abs(true_rho) <= np.abs(rhos)))+1)/(n_iter+1) # two-tailed test


    f,ax = plt.subplots()
    plt.hist(rhos,bins=20)
    ax.axvline(true_rho,  color = 'r', linestyle='--')
    ax.set(title=f"Permuted p: {perm_p:.3f}", ylabel="counts", xlabel="rho")
    plt.show()


def plot_mixed(df1, df2):
    """
    """
    # O-score VS Playlist
    fig, axs = plt.subplots(1,len(ROUNDS[1:]),sharey=True)
    for n, (att_round, session) in enumerate(zip(ROUNDS[1:], SESSIONS)):
        df_round = df1[df1.att_round == att_round]
        df_session = df2[df2.session.isin(session)]
        for group, c in zip(GROUPS, ["b","g"]):
            x = df_round[df_round.group == group].sort_values(by="PROLIFIC_PID").o_score.tolist()
            y = df_session[df_session.group == group].groupby(by='PROLIFIC_PID').playlist.sum().tolist()
            axs[n].scatter(x, y, label=group, s=50, c=c, marker='x')
            axs[n].legend(loc=2)
            axs[n].hlines(y=2.5, color='r', xmin=-1, xmax=6, linestyle='-')
            axs[n].vlines(x=2.5, color='r', ymin=-1, ymax=6, linestyle='-')
            axs[n].set_ylim(-0.5,5.5)
            axs[n].set_xlim(-0.5,5.5)
            axs[n].set_xlabel("O-score")
    axs[0].set_ylabel("Playlist visited")
    axs[0].set_title(label="Week 1")
    axs[1].set_title(label="Week 2")
    axs[2].set_title(label="Week 3")
    axs[3].set_title(label="Week 4")
    plt.show()

    # Like VS Playlist
    fig, axs = plt.subplots(1,len(ROUNDS[1:]),sharey=True)
    for n, (att_round, session) in enumerate(zip(ROUNDS[1:], SESSIONS)):
        df_round = df1[df1.att_round == att_round]
        df_session = df2[df2.session.isin(session)]
        for group, c in zip(GROUPS, ["b","g"]):
            x = df_session[df_session.group == group].groupby(by='PROLIFIC_PID').like.mean().tolist()
            y = df_session[df_session.group == group].groupby(by='PROLIFIC_PID').playlist.sum().tolist()
            axs[n].scatter(x, y, label=group, s=50, c=c, marker='x')
            axs[n].legend(loc=2)
            axs[n].hlines(y=2.5, color='r', xmin=-3, xmax=3, linestyle='-')
            axs[n].vlines(x=0, color='r', ymin=-1, ymax=6, linestyle='-')
            axs[n].set_ylim(-0.5,5.5)
            axs[n].set_xlim(-2.5,2.5)
            axs[n].set_xlabel("Like Mean Rating")
    axs[0].set_ylabel("Playlist visited")
    axs[0].set_title(label="Week 1")
    axs[1].set_title(label="Week 2")
    axs[2].set_title(label="Week 3")
    axs[3].set_title(label="Week 4")
    plt.show()

    # Like VS Familiarity
    fig, axs = plt.subplots(1,len(ROUNDS[1:]),sharey=True)
    for n, (att_round, session) in enumerate(zip(ROUNDS[1:], SESSIONS)):
        df_round = df1[df1.att_round == att_round]
        df_session = df2[df2.session.isin(session)]
        for group, c in zip(GROUPS, ["b","g"]):
            x = df_session[df_session.group == group].groupby(by='PROLIFIC_PID').like.mean().tolist()
            y = df_session[df_session.group == group].groupby(by='PROLIFIC_PID').familiarity.mean().tolist()
            axs[n].scatter(x, y, label=group, s=50, c=c, marker='x')
            axs[n].legend(loc=2)
            axs[n].hlines(y=0, color='r', xmin=-3, xmax=3, linestyle='-')
            axs[n].vlines(x=0, color='r', ymin=-3, ymax=3, linestyle='-')
            axs[n].set_ylim(-1.05,1.05)
            axs[n].set_xlim(-2.5,2.5)
            axs[n].set_xlabel("Like Mean Rating")    
    axs[0].set_ylabel("Familiarity Mean Rating")
    axs[0].set_title(label="Week 1")
    axs[1].set_title(label="Week 2")
    axs[2].set_title(label="Week 3")
    axs[3].set_title(label="Week 4")
    plt.show()

    # D-score VS Like
    fig, axs = plt.subplots(1,len(ROUNDS[1:]),sharey=True)
    for n, (att_round, session) in enumerate(zip(ROUNDS[1:], SESSIONS)):
        df_round = df1[df1.att_round == att_round]
        df_session = df2[df2.session.isin(session)]
        for group, c in zip(GROUPS, ["b","g"]):
            x = df_round[df_round.group == group].sort_values(by="PROLIFIC_PID").d_score.tolist()
            y = df_session[df_session.group == group].groupby(by='PROLIFIC_PID').like.mean().tolist()
            axs[n].scatter(x, y, label=group, s=50, c=c, marker='x')
            axs[n].legend(loc=2)
            axs[n].hlines(y=0, color='r', xmin=-1.5, xmax=1.5, linestyle='-')
            axs[n].vlines(x=0, color='r', ymin=-2.5, ymax=2.5, linestyle='-')
            axs[n].set_ylim(-2.5,2.5)
            axs[n].set_xlim(-1.5,1.5)
            axs[n].set_xlabel("D-score")
    axs[0].set_ylabel("Like Mean Rating")
    axs[0].set_title(label="Week 1")
    axs[1].set_title(label="Week 2")
    axs[2].set_title(label="Week 3")
    axs[3].set_title(label="Week 4")
    plt.show()


def plot_correlation(df):
    """
    """
    # Compute correlation matrices
    for att in ['d_score', 'o_score']:
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
                        for pid in df_join_att_g.PROLIFIC_PID.unique():
                            p1 = df_join_att_g[(df_join_att_g.att_round == att_round1) & (df_join_att_g.PROLIFIC_PID == pid)][att]
                            p2 = df_join_att_g[(df_join_att_g.att_round == att_round2) & (df_join_att_g.PROLIFIC_PID == pid)][att]

                            if len(p1) == 0 or len(p2) == 0: ## CHECK
                                s += 0
                            else:
                                p1 = p1.item()
                                p2 = p2.item()
                                s += ((p1-m1)/s1) * ((p2-m2)/s2)

                        s /= (len(df_join_att_g.PROLIFIC_PID.unique()) - 1)

                    CorrMatrix[c1,c2] = CorrMatrix[c2, c1] = s
            
            print(group, att)
            print(tabulate(CorrMatrix, headers=ROUNDS_LAB, tablefmt="github"))


    # Plot D-score
    fig, axs = plt.subplots(len(ROUNDS),len(ROUNDS),sharey=True,sharex=True)
    for group, c in zip(GROUPS, ["b","g"]):
        df_join_att_g = df[df.group == group]
        for c1, att_round1 in enumerate(ROUNDS):
            for c2, att_round2 in enumerate(ROUNDS):
                if c1 > c2:
                    continue
                elif c1 == c2:
                    axs[c1,c2].text(0,0, att_round1)
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
                    axs[c1,c2].plot(np.arange(-2,2), a*np.arange(-2,2)+b, c=c)  
                    axs[c1,c2].scatter(x,y,c=c)
                    axs[c1,c2].set_ylim(-1,1)
                    axs[c1,c2].set_xlim(-1,1)
                    axs[c1,c2].plot(np.arange(-2,2), np.arange(-2,2), c='r', linestyle='--')
                    axs[c2,c1].plot(np.arange(-2,2), a*np.arange(-2,2)+b, c=c)  
                    axs[c2,c1].scatter(x,y,c=c, label=group)
                    axs[c2,c1].set_ylim(-1,1)
                    axs[c2,c1].set_xlim(-1,1)
                    axs[c2,c1].plot(np.arange(-2,2), np.arange(-2,2), c='r', linestyle='--')
                    axs[c2,c1].legend(loc=2)

    plt.suptitle("D-score Correlation")
    plt.show()

    # Plot O-score
    fig, axs = plt.subplots(len(ROUNDS),len(ROUNDS),sharey=True,sharex=True)
    for group, c in zip(GROUPS, ["b","g"]):
        df_join_att_g = df[df.group == group]
        for c1, att_round1 in enumerate(ROUNDS):
            for c2, att_round2 in enumerate(ROUNDS):
                if c1 > c2:
                    continue
                elif c1 == c2:
                    axs[c1,c2].text(2.5,2.5, att_round1)
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
                    axs[c1,c2].plot(np.arange(-1,6), a*np.arange(-1,6)+b, c=c)    
                    axs[c1,c2].scatter(x,y,c=c)
                    axs[c1,c2].set_ylim(-0.5,5.5)
                    axs[c1,c2].set_xlim(-0.5,5.5)
                    axs[c1,c2].plot(np.arange(-1,6), np.arange(-1,6), c='r', linestyle='--')

                    axs[c2,c1].plot(np.arange(-1,6), a*np.arange(-1,6)+b, c=c)    
                    axs[c2,c1].scatter(x,y,c=c, label=group)
                    axs[c2,c1].set_ylim(-0.5,5.5)
                    axs[c2,c1].set_xlim(-0.5,5.5)
                    axs[c2,c1].plot(np.arange(-1,6), np.arange(-1,6), c='r', linestyle='--')

                    axs[c2,c1].legend(loc=2)
    plt.suptitle("O-score Correlation")
    plt.show()


def plot_ls(df):
    """
    """
    fig, axs = plt.subplots(2,4,sharey=True,sharex=True)
    for n, (group, c) in enumerate(zip(GROUPS, ["b","g"])):
        df_join_ls_group = df[df.group == group]
        df_join_pid = df_join_ls_group.groupby(by='PROLIFIC_PID')
        for m,(pid, df_pid) in enumerate(df_join_pid):
            axs[n,m].plot(range(1, 21), np.cumsum(df_pid.playlist), c=c, label=group)
            axs[n,m].set_ylim(0,15)
            axs[n,m].set_xlim(1,15)
            axs[n,m].set_title(pid)
            axs[n,m].set_yticks(np.arange(1, 21))
            axs[n,m].set_xticks(np.arange(1, 21))
            axs[n,m].grid()
            axs[n,m].legend(loc=2)
            axs[n,m].tick_params(axis="x", which="both", rotation=90)
    plt.suptitle('Playlist Interacted (cumulative)')
    plt.legend()
    plt.show()


    fig, axs = plt.subplots(2,4,sharey=True,sharex=True)
    for n, (group, c) in enumerate(zip(GROUPS, ["b","g"])):
        df_join_ls_group = df[df.group == group]
        df_join_pid = df_join_ls_group.groupby(by='PROLIFIC_PID')
        for m,(pid, df_pid) in enumerate(df_join_pid):
            axs[n,m].plot(range(1, 21), np.cumsum(df_pid.familiarity), c=c, label=group)
            axs[n,m].set_title(pid)
            axs[n,m].set_yticks(np.arange(-16, 9))
            axs[n,m].set_xticks(np.arange(1, 21))
            axs[n,m].grid()
            axs[n,m].legend(loc=2)
            axs[n,m].tick_params(axis="x", which="both", rotation=90)

    plt.suptitle('Track Familiarity (cumulative)')
    plt.legend()
    plt.legend()
    plt.show()


    fig, axs = plt.subplots(2,4,sharey=True,sharex=True)
    for n, (group, c) in enumerate(zip(GROUPS, ["b","g"])):
        df_join_ls_group = df[df.group == group]
        df_join_pid = df_join_ls_group.groupby(by='PROLIFIC_PID')
        for m,(pid, df_pid) in enumerate(df_join_pid):
            axs[n,m].plot(range(1, 21), np.cumsum(df_pid.like), c=c, label=group)
            axs[n,m].set_title(pid)
            axs[n,m].set_yticks(np.arange(-1, 24))
            axs[n,m].set_xticks(np.arange(1, 21))
            axs[n,m].grid()
            axs[n,m].legend(loc=2)
            axs[n,m].tick_params(axis="x", which="both", rotation=90)

    plt.suptitle('Track Liked (cumulative)')
    plt.legend()
    plt.show()    


if __name__ == "__main__":

    df_join_att = import_data("scores")
    df_join_att.d_score = - df_join_att.d_score
    df_join_cntx = import_data("cntx")
    df_join_ls = import_ls_data()
    df_join_ls = scale_ls_data(df_join_ls)


    # plot_scores(df_join_att)
    # plot_correlation(df_join_att)
    plot_cntx(df_join_cntx)
    # plot_mixed(df_join_att, df_join_ls)
    # plot_ls(df_join_ls)
    
