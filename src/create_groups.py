#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np


infile = "../data/prescreening/prescreening_compact.csv"

FEAT_DIFF = ['AVG_DK_SCORE','OPEN_SCORE', 'LIST_TIME', 'EM_LISTEN', 'TASTE_VARIED', 'EM_VARIED']
WEIGHTS = [0.35, 0.35, 0.1, 0.1, 0.1, 0.1]

if __name__ == "__main__":

    df = pd.read_csv(infile)
    artist_scores = []

    
    matches = []
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

        min_index, min_dist = min(dists, key=lambda p:p[1])
        
        matches.append(p1)
        matches.append(min_index)
        print(p1, min_index, min_dist)
