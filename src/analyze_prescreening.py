#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script takes as input the prescreening data from PsyToolKit
and computes the domain knowledge and openess score. As output,
it gives the input data plus the columns with the scores computed.
"""

import pandas as pd
import csv

ARTIST_SCORE = {"af:1":0.6,
                "af:2":0.15,
                "af:3":0.95,
                "af:4":0.4,
                "af:5":0.55,
                "af:6":0.5,
                "af:7":0.9,
                "af:8":0.65,
                "af:9":0.7,
                "af:10":0.3,
                "af:11":0.25,
                "af:12":0.85,
                "af:13":0.05,
                "af:14":0.2,
                "af:15":0.1,
                "af:16":0.75,
                "af:17":0.45,
                "af:18":0.35,
                "af:19":0.8,
                "af:20":1}

GENRE_SCORE = {"gf:1":0.05,
               "gf:2":0.8,
               "gf:3":0.5,
               "gf:4":0.4,
               "gf:5":0.25,
               "gf:6":0.3,
               "gf:7":1,
               "gf:8":0.1,
               "gf:9":0.55,
               "gf:10":0.85,
               "gf:11":0.65,
               "gf:12":0.45,
               "gf:13":0.2,
               "gf:14":0.75,
               "gf:15":0.6,
               "gf:16":0.7,
               "gf:17":0.95,
               "gf:18":0.35,
               "gf:19":0.15,
               "gf:20":0.9}

GUTTMAN = ["guttman:1",
           "guttman:2",
           "guttman:3",
           "guttman:4",
           "guttman:5"
           ]

infile = "../data/prescreening/data.csv"
outfile = "../data/prescreening/data_enriched.csv"

def get_weight(ans):
    """
    """
    if ans == 1:
        w = 1
    elif ans == 2:
        w = 0.5
    elif ans == 3:
        w = 0
    return w


if __name__ == "__main__":

    df = pd.read_csv(infile, delimiter='\t')
    artist_scores = []
    genre_scores = []
    avg_scores = []
    open_scores = []

    for part_id in df.PROLIFIC_PID.values:
        print(part_id)
        df_part = df[df.PROLIFIC_PID == part_id]
        # print(df_part)
        s_a = 0
        for k, v in ARTIST_SCORE.items():
            ans = df_part[k].item()
            w = get_weight(ans)
            s_a += w*v

        print(s_a, [df_part[k].item() for k,v in ARTIST_SCORE.items()])

        s_g = 0
        for k, v in GENRE_SCORE.items():
            ans = df_part[k].item()
            w = get_weight(ans)
            s_g += w*v

        print(s_g, [df_part[k].item() for k,v in GENRE_SCORE.items()])


        print((s_a + s_g)/2)
        print()

        s_o = 0
        for gut in GUTTMAN:
            ans = df_part[gut].item()
            if ans == 1:
                s_o +=1



        artist_scores.append("{:.2f}".format(s_a))
        genre_scores.append("{:.2f}".format(s_g))
        avg_scores.append("{:.2f}".format((s_a + s_g)/2))
        open_scores.append(s_o)


    df = df.assign(artist_fam_score=artist_scores,
                   genre_fam_score=genre_scores,
                   avg_fam_score=avg_scores,
                   open_score=open_scores)

    df.to_csv(outfile, index=False)

    print("\nFound {} participants".format(len(df.PROLIFIC_PID.values)))
