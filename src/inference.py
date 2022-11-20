#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg

from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pingouin import ancova, ttest, wilcoxon, mwu, cronbach_alpha, corr


ATT_FOLDER = "../data/attitudes"
LS_FOLDER = "../data/ls"

INFILE_1 = "../data/listenbrainz/results/logs_analysis.csv"
INFILE_2 = "../data/listenbrainz/results/logs_diff_analysis.csv"

ROUNDS =  ["00", "01", "02", "03", "04", "10"]
ROUNDS_LAB = ['Pre', 'Week 1', "Week 2", "Week 3", "Week 4", "Post"]
GROUPS = ["HD", "LD"]

COLS = ["username", "group", "phase"]

COLS_VAR = ["EM_genres_unique_p", "EM_genres_count_p",
            "EM_artists_unique_p", "EM_artists_count_p",
            "EM_tracks_unique_p", "EM_tracks_count_p"]

# COLS_VAR = ["gini_genres", "gini_artists", "gini_tracks",
#         "gini_EM_genres","gini_EM_artists","gini_EM_tracks"]

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
    df_merge = df[df.phase == 'PRE'].merge(df[df.phase == 'POST'], on=('username', 'group'))

    df_merge['group'].mask(df_merge['group'] == 'LD', 0, inplace=True)
    df_merge['group'].mask(df_merge['group'] == 'HD', 1, inplace=True)

    df_merge.dropna(inplace=True)

    for col in COLS_VAR:

        print("\n\nPre-post Follow-up: {}".format(col))
        y = col + "_y"
        x = col + "_x"
        df_merge['diff'] = df_merge[y]
        dv = 'diff'
        between = "group"

        formula = "{} ~ C({})".format(dv, between)
        model = smf.ols(formula, data=df_merge).fit()
        print(model.summary())

        print("\n\nPre-post Change Analysis: {}".format(col))
        y = col + "_y"
        x = col + "_x"
        df_merge['diff'] = df_merge[y] - df_merge[x]
        dv = 'diff'
        between = "group"

        formula = "{} ~ C({})".format(dv, between)
        model = smf.ols(formula, data=df_merge).fit()
        print(model.summary())


def pre_post_analysis_att(t1, t2):
    """
    """
    df_merge = df_join_att[df_join_att.att_round == t1].merge(df_join_att[df_join_att.att_round == t2], on=('PROLIFIC_PID', 'group'))
    df_merge['group'].mask(df_merge['group'] == 'LD', 0, inplace=True)
    df_merge['group'].mask(df_merge['group'] == 'HD', 1, inplace=True)

    # PRE-POST ANALYSIS ANCOVA
    for col in ['d_score', 'o_score']:

        print("\n\nPre-post Follow-up: {}".format(col))
        y = col + "_y"
        x = col + "_x"
        df_merge['diff'] = df_merge[y]
        dv = 'diff'
        between = "group"

        formula = "{} ~ C({})".format(dv, between)
        model = smf.ols(formula, data=df_merge).fit()
        print(model.summary())

        print("\n\nPre-post Change Analysis: {}".format(col))
        y = col + "_y"
        x = col + "_x"
        df_merge['diff'] = df_merge[y] - df_merge[x]
        dv = 'diff'
        between = "group"

        formula = "{} ~ C({})".format(dv, between)
        model = smf.ols(formula, data=df_merge).fit()
        print(model.summary())


def old_prepost():
        ## Pre-Post analysis
    # pre_post_analysis_logs()

    # for times in [("00", "01", "two-sided"), ("01", "04","greater"), ("04", "10", "two-sided"), ("00", "10", "greater")]:
    #     print()
    #     print(times)
    #     t1, t2, alt = times
    #     pre_post_analysis_att(t1,t2)



    # ### Gaussian -- Exchangeable 
    # exc = sm.cov_struct.Exchangeable()
    # mod1 = smf.gee("d_score ~ 0 + att_round + group + group * att_round",
    #                "PROLIFIC_PID",
    #                data=df_join_att, 
    #                cov_struct=exc)
    # res1 = mod1.fit()
    # print(res1.summary())

    # ### Ordinal -- Exchangeable
    # model = smf.ordinal_gee("o_score ~ 0 + att_round + group + group * att_round",
    #                         "PROLIFIC_PID",
    #                         data=df_join_att, 
    #                         cov_struct=exc)
    # result = model.fit_regularized(pen_wt=0.00)
    # print(result.summary())


    ############################




    # df_join_att['att_round'] = df_join_att['att_round'].astype(float)
    # df_join_att['group'] = df_join_att['group'].astype(float)
    # df_join_att['o_score'] = df_join_att['o_score'].astype(float)


    # md = smf.mixedlm("d_score ~ group + group * att_round", df_join_att, groups="group")
    # mdf = md.fit()
    # print(mdf.summary())

    # fam = sm.families.Gaussian()
    # ind = sm.cov_struct.Exchangeable()
    # mod = smf.gee("d_score ~ group + group * att_round", "PROLIFIC_PID", df_join_att,
    #               cov_struct=ind, family=fam)
    # res = mod.fit()
    # print(res.summary())

    # re = mdf.random_effects
    # # Multiply each BLUP by the random effects design matrix for one group
    # rex = [np.dot(md.exog_re_li[j], re[k]) for (j, k) in enumerate(md.group_labels)]
    # # Add the fixed and random terms to get the overall prediction
    # rex = np.concatenate(rex)
    # yp = mdf.fittedvalues + rex


    # md = smf.mixedlm("o_score ~ att_round + group + group * att_round", df_join_att, groups=df_join_att["group"])
    # mdf = md.fit()
    # print(mdf.summary())

    # fam = sm.families.Gaussian()
    # ind = sm.cov_struct.Exchangeable()
    # mod = smf.gee("d_score ~ group + group * att_round", "PROLIFIC_PID", df_join_att,
    #               cov_struct=ind, family=fam)
    # res = mod.fit()
    # print(res.summary())





    # fam = sm.families.Binomial()
    # ind = sm.cov_struct.Autoregressive()
    # mod = smf.ordinal_gee("o_score ~ 0 + att_round + group + group * att_round", "PROLIFIC_PID", df_join_att,
    #                       cov_struct=ind, family=fam)
    # res = mod.fit()
    # print(res.summary())




    # ### Autoregressive 
    # fam = sm.families.Gaussian()
    # ind = sm.cov_struct.Autoregressive()
    # times = (df_join_att['att_round'].values)
    # mod = smf.gee("d_score ~ 0 + att_round + group + group * att_round", "PROLIFIC_PID", df_join_att,
    #               cov_struct=ind, family=fam, time=times)
    # res = mod.fit()
    # print(res.summary())

    # ### Ordinal -- OddsRatio
    # gor = sm.cov_struct.GlobalOddsRatio("ordinal")
    # model = smf.ordinal_gee("o_score ~ 0 + att_round + group + group * att_round", df_join_att["group"], df_join_att,
    #                             cov_struct=gor)
    # result = model.fit()
    # print(result.summary())


    # ### Gaussian -- Exchangeable 
    # fam = sm.families.Gaussian()
    # ind = sm.cov_struct.Exchangeable()
    # times = (df_join_att['att_round'].values)
    # mod2 = smf.gee("o_score ~ 0 + att_round + group + group * att_round",
    #               "PROLIFIC_PID",
    #               data=df_join_att,
    #               family=fam,
    #               cov_struct=ind, 
    #               time=times)
    # res2 = mod2.fit()
    # print(res2.summary())

    # # Format data
    # df_join_att['att_round'].mask(df_join_att['att_round'] == '00', 1, inplace=True)
    # df_join_att['att_round'].mask(df_join_att['att_round'] == '01', 2, inplace=True)
    # df_join_att['att_round'].mask(df_join_att['att_round'] == '02', 3, inplace=True)
    # df_join_att['att_round'].mask(df_join_att['att_round'] == '03', 4, inplace=True)
    # df_join_att['att_round'].mask(df_join_att['att_round'] == '04', 5, inplace=True)
    # df_join_att['att_round'].mask(df_join_att['att_round'] == '10', 6, inplace=True)

    df_join_att['o_score'].mask(df_join_att['o_score'] == 0, 0, inplace=True)
    df_join_att['o_score'].mask(df_join_att['o_score'] == 1, 0, inplace=True)
    df_join_att['o_score'].mask(df_join_att['o_score'] == 2, 0, inplace=True)
    df_join_att['o_score'].mask(df_join_att['o_score'] == 3, 1, inplace=True)
    df_join_att['o_score'].mask(df_join_att['o_score'] == 4, 1, inplace=True)
    df_join_att['o_score'].mask(df_join_att['o_score'] == 5, 1, inplace=True)


    df_join_att['group'].mask(df_join_att['group'] == 'LD', 0, inplace=True)
    df_join_att['group'].mask(df_join_att['group'] == 'HD', 1, inplace=True)

    pre_post_analysis_att("00","10")


def wilcoxon_scores():
    """
    """
    # Wilcoxon whole population
    for times in [("00", "01", "two-sided"), ("00", "04","greater"), ("04", "10", "two-sided"), ("00", "10", "greater")]:
        print()
        print(times)
        t1, t2, alt = times
        # Scores
        d_0 = []
        d_1 = [] 
        o_0 = []
        o_1 = []
        for pid in df_join_att.PROLIFIC_PID.unique():
            # print(pid)
            id_0 = df_join_att[(df_join_att.att_round == t1) & (df_join_att.PROLIFIC_PID == pid)].d_score.values
            id_1 = df_join_att[(df_join_att.att_round == t2) & (df_join_att.PROLIFIC_PID == pid)].d_score.values
            io_0 = df_join_att[(df_join_att.att_round == t1) & (df_join_att.PROLIFIC_PID == pid)].o_score.values
            io_1 = df_join_att[(df_join_att.att_round == t2) & (df_join_att.PROLIFIC_PID == pid)].o_score.values


            if id_0.size != 0 and id_1.size != 0 and io_0.size != 0 and io_1.size != 0:
                d_0.append(id_0[0])
                d_1.append(id_1[0])
                o_0.append(io_0[0])
                o_1.append(io_1[0])
            

        print(len(d_0), len(d_1), np.median(d_0), np.median(d_1))
        print(wilcoxon(d_0, d_1, alternative=alt))
        df = pd.DataFrame({'Q1': d_0, 'Q2': d_1})
        # print(cronbach_alpha(df))
        print(corr(d_0, d_1))

        if alt == 'greater':
            alt = 'less'

        print(len(o_0), len(o_1), np.median(o_0), np.median(o_1))
        print(wilcoxon(o_0, o_1, alternative=alt))
        df = pd.DataFrame({'Q1': o_0, 'Q2': o_1})
        # print(cronbach_alpha(df))
        print(corr(o_0, o_1))


def IQR(data):
    """
    """
    q3, q1 = np.percentile(data, [75 ,25])
    iqr = q3 - q1
    return iqr

if __name__ == "__main__":


    # Import data
    df_join_att = import_data("scores")
    df_join_att.d_score = - df_join_att.d_score
    df_join_att = filter_dataframe(df_join_att, 'inc', 'pid')
    df_join_att_HD = df_join_att[df_join_att.group == 'HD']
    df_join_att_LD = df_join_att[df_join_att.group == 'LD']

    df_join_ls = import_ls_data()
    df_join_ls = scale_ls_data(df_join_ls)


    df_join_cntx = import_data("cntx")
    df_join_cntx = filter_dataframe(df_join_cntx, 'inc', 'pid')
    df_join_cntx_HD = df_join_cntx[df_join_cntx.group == 'HD']
    df_join_cntx_LD = df_join_cntx[df_join_cntx.group == 'LD']


    COLUMNS = ['Relaxing', 'Commuting', 'Partying', 'Running', 
               'Shopping', 'Sleeping', 'Studying', 'Working', 'Tempo', 'Danceability',
               'Acousticness', 'Instrumentalness', 'Gender', 'Skin', 'Origin', 'Age']

    TIMES = [("00", "04","greater"), ("00", "10", "greater")]


    # # Compare Contexts PRE-POST
    # for df in [df_join_cntx, df_join_cntx_HD, df_join_cntx_LD]:
    #     for times in TIMES:
    #         print()
    #         print(times)
    #         t1, t2, alt = times

    #         all_values1 = []
    #         all_values2 = []
    #         for pid in df.PROLIFIC_PID.unique():
    #             skip = False
    #             pid_values1, pid_values2 = [], []
    #             for column in COLUMNS:
    #                 value1 = df[(df.att_round == t1) & (df.PROLIFIC_PID == pid)][column].values
    #                 value2 = df[(df.att_round == t2) & (df.PROLIFIC_PID == pid)][column].values

    #                 if value1.size > 0 and value2.size > 0:
    #                     pid_values1.append(value1[0])
    #                     pid_values2.append(value2[0])
    #                 else:
    #                     skip = True

    #             if not skip:
    #                 all_values1.append(pid_values1)
    #                 all_values2.append(pid_values2)
    #                 # print (pid, "1", pid_values1)
    #                 # print (pid, "2", pid_values2)
                

    #         for n,col in enumerate(COLUMNS):
    #             l1 = [item[n] for item in all_values1]
    #             l2 = [item[n] for item in all_values2]

    #             print("#########", col)
    #             print(len(l1), len(l2), np.median(l1), IQR(l1), np.median(l2), IQR(l2))
    #             print(wilcoxon(l1,l2))
    #             print(corr(l1,l2))



  # Compare context HD and LD
    for t in ['00', '04','10']:
        print()
        print("####",t)
        for column in COLUMNS:
            col_HD = df_join_cntx_HD[(df_join_cntx_HD.att_round == t)][column].tolist()
            col_LD = df_join_cntx_LD[(df_join_cntx_LD.att_round == t)][column].tolist()
            print("####", column)
            print(mwu(col_HD, col_LD))
            
