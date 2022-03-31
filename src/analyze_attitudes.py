#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt

ATT_FOLDER = "../data/attitudes"

CONTEXTS = ["Relaxing", "Commuting", "Partying", "Running","Shopping",
            "Sleeping", "Studying", "Working"]
TRACK_FEATS = ["Tempo", "Danceability", "Acousticness", "Instrumentalness"]
ARTIST_FEATS = ["Gender", "Skin", "Origin", "Age"]
ROUNDS =  ["00", "01", "02"]
GROUPS = ["g1", "g2"]


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


def plot_scores():
    """
    """

    # Plot D-score
    for group, c in zip(GROUPS, ["b","g"]):
        df_group = df_join_att[df_join_att.group == group]

        for pid in df_group.PROLIFIC_PID.unique():
            x = [1,2,3]
            y = [df_group[(df_group.PROLIFIC_PID == pid) & (df_group.att_round == att_round)].d_score.item() for att_round in ROUNDS]
            plt.plot(x, y, c=c, label=group)

    plt.title('D-score')
    plt.xticks(x,['Pre', '1 week', "2 week"], horizontalalignment='right')
    plt.ylim([-1,1])
    plt.legend()
    plt.grid()
    plt.show()

    # Plot O-score
    for group, c in zip(GROUPS, ["b","g"]):
        df_group = df_join_att[df_join_att.group == group]

        for pid in df_group.PROLIFIC_PID.unique():
            x = range(3)
            y = [df_group[(df_group.PROLIFIC_PID == pid) & (df_group.att_round == att_round)].o_score.item() for att_round in ROUNDS]
            plt.plot(x, y, c=c, label=group)

    plt.title('O-score')
    plt.xticks(x,['Pre', '1 week', "2 week"], horizontalalignment='right')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot o-score VS d-score
    fig, axs = plt.subplots(1,3,sharey=True)
    for n, att_round in enumerate(ROUNDS):
        df_round = df_join_att[df_join_att.att_round == att_round]
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
    axs[1].set_title(label="01")
    axs[2].set_title(label="02")
    plt.show()



if __name__ == "__main__":

    df_join_att = import_data("scores")
    df_join_cntx = import_data("cntx")

    plot_scores() 


    fig, axs = plt.subplots(2,4, sharex=True, sharey=True)
    axs = axs.reshape(-1)
    for column, ax in zip(CONTEXTS, axs):
        bp_dict = df_join_cntx.boxplot(by=['group','att_round'],
                                       column=column,
                                       layout=(2,2),
                                       return_type='both',
                                       patch_artist = True,
                                       ax=ax)
        ax.set_yticklabels(['Disagree', '', '', '', 'Agree'])
        ax.set_yticks([1,2,3,4,5])
        ax.set_xlabel('')
        colors = ['r','b', 'b', 'r', 'y', 'y']
        for row_key, (ax,row) in bp_dict.iteritems():
            for i, box in enumerate(row['boxes']):
                box.set_facecolor(colors[i])
        plt.suptitle("In which contexts would you listen to Electronic Music?")
    plt.show()



    fig, axs = plt.subplots(2,2, sharex=True)
    axs = axs.reshape(-1)
    for column, ax in zip(TRACK_FEATS, axs):
        bp_dict = df_join_cntx.boxplot(by=['group','att_round'],
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
        colors = ['r','b', 'b', 'r', 'y', 'y']
        for row_key, (ax,row) in bp_dict.iteritems():
            for i, box in enumerate(row['boxes']):
                box.set_facecolor(colors[i])
        plt.suptitle("Which features do you associate Electronic Music?")
    plt.show()



    fig, axs = plt.subplots(2,2, sharex=True)
    axs = axs.reshape(-1)
    for column, ax in zip(ARTIST_FEATS, axs):
        bp_dict = df_join_cntx.boxplot(by=['group','att_round'],
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
        colors = ['r','b', 'b', 'r', 'y', 'y']
        for row_key, (ax,row) in bp_dict.iteritems():
            for i, box in enumerate(row['boxes']):
                box.set_facecolor(colors[i])
        plt.suptitle("Which characteristics do you associate Electronic Music artists?")
    plt.show()

