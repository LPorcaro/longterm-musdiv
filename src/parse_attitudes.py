#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd
import argparse

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

ARTIST_FEATS = {"att1:1": "Gender",
                "att1:2": "Skin",
                "att1:3": "Origin",
                "att1:4": "Age"
                }

IAT_FOLDER = "../data/iat"
ATT_FOLDER = "../data/attitudes"

ROUNDS =  ["00", "01", "02", "03", "04", "10"]

GROUPS = ["HD", "LD"]

def analyze_iat(iat_path):
    """
    """
    infile = os.path.join(IAT_FOLDER, att_round, iat_path)

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


def parse(att_round, group):
    """
    """
    group_infile = "{}_{}.csv".format(group, att_round)
    infile = os.path.join(ATT_FOLDER, att_round, group_infile)

    group_dscore_outf = "{}_{}_scores.csv".format(group, att_round)
    outfile1 = os.path.join(ATT_FOLDER, att_round, group_dscore_outf)
    group_cntx_outf = "{}_{}_cntx.csv".format(group, att_round)
    outfile2 = os.path.join(ATT_FOLDER, att_round, group_cntx_outf)

    df = pd.read_csv(infile, delimiter='\t')

    # Extract Context 
    with open(outfile2, "w+") as outf:
        _writer = csv.writer(outf)

        header = ["PROLIFIC_PID"]
        header += list(CONTEXTS.values())
        header += list(TRACK_FEATS.values())
        header += list(ARTIST_FEATS.values())

        _writer.writerow(header)

        for idx in range(len(df)):
            row = [df.iloc[idx].PROLIFIC_PID]
            row += [df.iloc[idx][item] for item in CONTEXTS]
            row += [df.iloc[idx][item] for item in TRACK_FEATS]
            row += [df.iloc[idx][item] for item in ARTIST_FEATS]
            _writer.writerow(row)


    # Compute d-score and o-score
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
    df_out = df_out.assign(PROLIFIC_PID=df.PROLIFIC_PID.values,
                           d_score=dscores,
                           o_score=open_scores)

    df_out.to_csv(outfile1, index=False)


def arg_parser():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--group", type=str, dest='group_name',
                        help="Group name")
    parser.add_argument("-r", "--round", type=str, dest='round_name',
                        help="Round name")
    parser.add_argument("-a", "--all", action='store_true', dest='all',
                        help="Parse all")
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = arg_parser()

    if not args.all:
        att_round = args.round_name
        group = args.group_name
        parse(att_round, group)
    else:
        for att_round in ROUNDS:
            for group in GROUPS:
                parse(att_round, group)

