#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

infile = "../data/prescreening/selected_part_comp.csv"

FEAT_DIFF = ['AVG_DK_SCORE','O_SCORE', 'D_SCORE', 'EM_tracks_count']
WEIGHTS = [0.25, 0.25, 0.25, 0.25]

if __name__ == "__main__":

    df = pd.read_csv(infile)
    df[FEAT_DIFF] = (df[FEAT_DIFF]-df[FEAT_DIFF].mean())/df[FEAT_DIFF].std()
    artist_scores = []

    dist_tot = []
    matches = []
    HD = set()
    LD = set()
    # for p1 in range(len(df)-1,0,-1):
    for p1 in range(len(df)):
        dists = []
        if p1 in matches:
            continue
        for p2 in range(len(df)):
            if p1 == p2:
                continue
            elif p2 in matches:
                continue

            v1 = df.loc[p1][FEAT_DIFF].tolist()
            v2 = df.loc[p2][FEAT_DIFF].tolist()
            d = [abs(j-i) for i, j in zip(v1, v2)]
            d = sum([j*i for i, j in zip(d, WEIGHTS)])
            dists.append((p2,d))

        if dists:
            min_index, min_dist = min(dists, key=lambda p:p[1])
            matches.append(p1)
            matches.append(min_index)
            print(p1, min_index, round(min_dist,2))
            dist_tot.append(min_dist)
            HD.add(p1)
            LD.add(min_index)
        else:
            print(p1,p2)


    print(np.mean(dist_tot))

    # print(df.iloc[list(HD)].PROLIFIC_PID.tolist())
    # print(df.iloc[list(LD)].PROLIFIC_PID.tolist())

    for pid in df.PROLIFIC_PID.tolist():
        if pid in df.iloc[list(HD)].PROLIFIC_PID.tolist():
            continue
        elif pid in df.iloc[list(LD)].PROLIFIC_PID.tolist():
            continue
        else:
            print(pid)