#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import csv

ARTIST_SCORE = {"af1:1":0.6,
                "af1:2":0.15,
                "af1:3":0.95,
                "af1:4":0.4,
                "af1:5":0.55,
                "af1:6":0.5,
                "af1:7":0.9,
                "af1:8":0.65,
                "af1:9":0.7,
                "af1:10":0.3,
                "af1:11":0.25,
                "af1:12":0.85,
                "af1:13":0.05,
                "af1:14":0.2,
                "af1:15":0.1,
                "af1:16":0.75,
                "af1:17":0.45,
                "af1:18":0.35,
                "af1:19":0.8,
                "af1:20":1}

GENRE_SCORE = {"gf1:1":0.05,
               "gf1:2":0.8,
               "gf1:3":0.5,
               "gf1:4":0.4,
               "gf1:5":0.25,
               "gf1:6":0.3,
               "gf1:7":1,
               "gf1:8":0.1,
               "gf1:9":0.55,
               "gf1:10":0.85,
               "gf1:11":0.65,
               "gf1:12":0.45,
               "gf1:13":0.2,
               "gf1:14":0.75,
               "gf1:15":0.6,
               "gf1:16":0.7,
               "gf1:17":0.95,
               "gf1:18":0.35,
               "gf1:19":0.15,
               "gf1:20":0.9}


infile = "../data/prescreening/data.csv"
outfile = "../data/prescreening/score_participants.csv"


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

    df = pd.read_csv(infile)

    with open(outfile, 'w+') as outf:
        _writer = csv.writer(outf)
        _writer.writerow(['pid', 'artist score', 'genre score', 
                          'average score'])
        for part_id in df.participant.values:
            print(part_id)
            df_part = df[df.participant == part_id]

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
            _writer.writerow([part_id, s_a, s_g, (s_a + s_g)/2])
