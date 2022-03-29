#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd
from datetime import datetime

now = datetime.now()
date_time = now.strftime("%Y%m%d")

CONTEXTS = {"af1:1":"Relaxing",
            "af1:2":"Commuting",
            "af1:3":"Partying",
            "af1:4":"Running",
            "af1:5":"Shopping",
            "af1:6":"Sleeping",
            "af1:7":"Studying",
            "af1:8":"Working",
            }

GUTTMAN = ["guttman:1",
           "guttman:2",
           "guttman:3",
           "guttman:4",
           "guttman:5"
           ]

TRACK_FEATS = {"att0:1": "Tempo",
               "att0:2": "Danceability",
               "att0:3": "Acousticness",
               "att0:4": "Instrumentalness"
               }

ARTIS_FEATS = {"att1:1": "Gender",
               "att1:2": "Skin",
               "att1:3": "Origin",
               "att1:4": "Age"
               }


IAT_FOLDER = "../data/iat/1st"
ATT_FILE = "../data/attitudes/1st/g1_1st.csv"
ATT_FILE_P_OUT = "../data/attitudes/1st/g1_1st_part_{}.csv".format(date_time)
ATT_FILE_F_OUT = "../data/attitudes/1st/g1_1st_feat_{}.csv".format(date_time)


def analyze_iat(infile):
    """
    """
    infile = os.path.join(IAT_FOLDER, infile)

    df = pd.DataFrame(columns = ["block", "block_num", "item", "rt", "status"])

    with open(infile, 'r') as inf:
        _reader = csv.reader(inf, delimiter=" ")

        for row in _reader:
            block, block_num, item, rt, status = row

            if block_num == "2" or block_num == "4":
                continue
            elif int(rt) < 350:
                continue
            elif int(rt) >= 3000:
                continue

            df.loc[len(df)] = [block, block_num, item, int(rt), int(status)]

    pen_left = round(df[df.block == "mix_compatible"].rt.mean() + 400)
    pen_right = round(df[df.block == "mix_incompatible"].rt.mean() + 400)

    df.loc[(df.status == 2) & (df.block == "mix_compatible"), 'rt'] = pen_left
    df.loc[(df.status == 2) & (df.block == "mix_incompatible"), 'rt'] = pen_right

    avg_left = df[df.block == "mix_compatible"].rt.mean()
    avg_right = df[df.block == "mix_incompatible"].rt.mean()
    all_std = df[df.status == 1].rt.std()

    dscore = (avg_left - avg_right) / all_std

    return round(dscore, 3)




if __name__ == "__main__":

    df = pd.read_csv(ATT_FILE, delimiter='\t')


    with open(ATT_FILE_F_OUT, "w+") as outf:
        _writer = csv.writer(outf)
        _writer.writerow(["Feature", "Median", "IQR"])
    
        for item in CONTEXTS:
            count, mean, std, _min, q1, q2, q3, _max = df[item].describe()
            _writer.writerow([CONTEXTS[item], q2, q3-q1])

        
        for item in TRACK_FEATS:
            count, mean, std, _min, q1, q2, q3, _max = df[item].describe()
            _writer.writerow([TRACK_FEATS[item], q2, q3-q1])

        
        for item in ARTIS_FEATS:
            count, mean, std, _min, q1, q2, q3, _max = df[item].describe()
            _writer.writerow([ARTIS_FEATS[item], q2, q3-q1])


    open_scores = []
    dscores = []
    for part_id in df.PROLIFIC_PID.values:
        df_part = df[df.PROLIFIC_PID == part_id]
        
        s_o = 0
        for gut in GUTTMAN:
            ans = df_part[gut].item()
            if ans == 1:
                s_o +=1

        open_scores.append(s_o)

        dscores.append(analyze_iat(df_part["testexperiment:1"].item()))


    df_out = pd.DataFrame()
    df_out = df_out.assign(prolific_pid=df.PROLIFIC_PID.values,
                           d_scores=dscores,
                           openess_score=open_scores)

    df_out.to_csv(ATT_FILE_P_OUT, index=False)