#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant

from pingouin import ancova

ATT_FOLDER = "../data/attitudes"
LS_FOLDER = "../data/ls"

INFILE_1 = "../data/listenbrainz/results/logs_analysis.csv"
INFILE_2 = "../data/listenbrainz/results/logs_diff_analysis.csv"

ROUNDS =  ["00", "01", "02", "03", "04", "10"]
ROUNDS_LAB = ['Pre', 'Week 1', "Week 2", "Week 3", "Week 4", "Post"]
GROUPS = ["HD", "LD"]

COLS = ["username", "group", "phase"]

# COLS_VAR = ["EM_genres_unique_p", "EM_genres_count_p",
#             "EM_artists_unique_p", "EM_artists_count_p",
#             "EM_tracks_unique_p", "EM_tracks_count_p"]

COLS_VAR = ["gini_genres", "gini_artists", "gini_tracks",
        "gini_EM_genres","gini_EM_artists","gini_EM_tracks"]

# COLS_VAR = ["genres_unique","artists_unique","tracks_unique",
#         "EM_genres_unique", "EM_artists_unique", "EM_tracks_unique"]

# COLS_VAR = ["genres_count", "artists_count", "tracks_count",
#         "EM_genres_count", "EM_artists_count", "EM_tracks_count"]

fsize = 20


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


def pre_post_analysis_logs():
    """
    """
    df = pd.read_csv(INFILE_1)
    df = df[COLS + COLS_VAR]
    df = filter_dataframe(df, 'inc', 'uname')
    # PRE-POST ANALYSIS ANCOVA
    df_merge = df[df.phase == 'PRE'].merge(df[df.phase == 'POST'], on=('username', 'group'))

    df_merge['group'].mask(df_merge['group'] == 'LD', 0, inplace=True)
    df_merge['group'].mask(df_merge['group'] == 'HD', 1, inplace=True)

    df_merge.dropna(inplace=True)

    for col in COLS_VAR:
        print("\n\nPre-post: {}".format(col))
        dv = col + "_y"
        covar = col + "_x"
        between = "group"

        formula = "{} ~ C({}) + {}".format(dv, between, covar)
        model = smf.ols(formula, data=df_merge).fit()
        print(model.summary())
        print(ancova(data=df_merge, dv=dv, covar=covar, between=between))

        # # Plot
        # fig, ax = plt.subplots()
        # x_LD = df_merge[(df_merge.group == 0)][dv]
        # y_LD = df_merge[(df_merge.group == 0)][covar]
        # x_HD = df_merge[(df_merge.group == 1)][dv]
        # y_HD = df_merge[(df_merge.group == 1)][covar]

        # frg = 2

        # ax.scatter(x_LD, y_LD, color='r', marker='x', label='LD')
        # ax.scatter(x_HD, y_HD, color='g', marker='x', label='HD')
        # a, b = np.polyfit(x_LD, y_LD, 1)
        # ax.plot(np.arange(frg), a*np.arange(frg)+b,color='r', linestyle='--')  
        # a, b = np.polyfit(x_HD, y_HD, 1)
        # ax.plot(np.arange(frg)-0.3, a*np.arange(frg)+b,color='g', linestyle='--') 

        # ax.set_xlabel(dv, fontsize = fsize)
        # ax.set_ylabel(covar, fontsize = fsize)
        # ax.set_title('', fontsize = fsize)
        # plt.legend(fontsize = fsize)
        # plt.grid()
        # plt.show() 

def pre_post_analysis_att():
    """
    """
    df = pd.read_csv(INFILE_1)
    df = df[COLS + COLS_VAR]

    # PRE-POST ANALYSIS ANCOVA
    df_merge = df_join_att[df_join_att.att_round == "00"].merge(df_join_att[df_join_att.att_round == "10"], on=('PROLIFIC_PID', 'group'))
    df_merge['group'].mask(df_merge['group'] == 'LD', 0, inplace=True)
    df_merge['group'].mask(df_merge['group'] == 'HD', 1, inplace=True)

    for frg, col in zip([2,6],['d_score', 'o_score']):
        print("\n\nPre-post: {}".format(col))
        dv = col + "_y"
        covar = col + "_x"
        between = "group"

        formula = "{} ~ C({}) + {}".format(dv, between, covar)
        model = smf.ols(formula, data=df_merge).fit()
        # print(model.summary())
        print(ancova(data=df_merge, dv=dv, covar=covar, between=between))


        # fig, ax = plt.subplots()
        # x_LD = df_merge[(df_merge.group == 0)][covar]
        # y_LD = df_merge[(df_merge.group == 0)][dv]
        # x_HD = df_merge[(df_merge.group == 1)][covar]
        # y_HD = df_merge[(df_merge.group == 1)][dv]

        # ax.scatter(x_LD, y_LD, color='r', marker='x', label='LD')
        # ax.scatter(x_HD, y_HD, color='g', marker='x', label='HD')
        # a, b = np.polyfit(x_LD, y_LD, 1)
        # ax.plot(np.arange(frg), a*np.arange(frg)+b,color='r', linestyle='--')  
        # a, b = np.polyfit(x_HD, y_HD, 1)
        # ax.plot(np.arange(frg)-0.3, a*np.arange(frg)+b,color='g', linestyle='--') 

        # ax.set_xlabel(covar, fontsize = fsize)
        # ax.set_ylabel(dv, fontsize = fsize)
        # ax.set_title('', fontsize = fsize)
        # plt.legend(fontsize = fsize)
        # plt.grid()
        # plt.show() 


if __name__ == "__main__":




    df_join_att = import_data("scores")
    df_join_att.d_score = - df_join_att.d_score

    df_join_att = filter_dataframe(df_join_att, 'full', 'pid')
    pre_post_analysis_att()


    # df_join_cntx = import_data("cntx")
    # df_join_ls = import_ls_data()
    # df_join_ls = scale_ls_data(df_join_ls)

    # pre_post_analysis_logs()

    






    # df_join_att['att_round'].mask(df_join_att['att_round'] == '00', 0, inplace=True)
    # df_join_att['att_round'].mask(df_join_att['att_round'] == '01', 1, inplace=True)
    # df_join_att['att_round'].mask(df_join_att['att_round'] == '02', 1, inplace=True)
    # df_join_att['att_round'].mask(df_join_att['att_round'] == '03', 1, inplace=True)
    # df_join_att['att_round'].mask(df_join_att['att_round'] == '04', 1, inplace=True)
    # df_join_att['att_round'].mask(df_join_att['att_round'] == '10', 1, inplace=True)
    # df_join_att['group'].mask(df_join_att['group'] == 'LD', 0, inplace=True)
    # df_join_att['group'].mask(df_join_att['group'] == 'HD', 1, inplace=True)
    # df_join_att['att_round'] = df_join_att['att_round'].astype(float)
    # df_join_att['group'] = df_join_att['group'].astype(float)
    # df_join_att['o_score'] = df_join_att['o_score'].astype(float)


    # fam = sm.families.Gaussian()
    # ind = sm.cov_struct.Autoregressive()
    # mod = smf.gee("d_score ~ att_round + group + group * att_round", "PROLIFIC_PID", df_join_att,
    #               cov_struct=ind, family=fam)
    # res = mod.fit()
    # print(res.summary())


    # fam = sm.families.Binomial()
    # ind = sm.cov_struct.Exchangeable()
    # mod = smf.ordinal_gee("o_score ~ 0 + att_round + group + group * att_round", "PROLIFIC_PID", df_join_att,
    #                       cov_struct=ind, family=fam)
    # res = mod.fit()
    # print(res.summary())



