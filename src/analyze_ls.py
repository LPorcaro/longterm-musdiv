#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt

LS_FOLDER = "../data/ls"
GROUPS = ["g1", "g2"]


def import_data():
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

if __name__ == "__main__":

    df_join = import_data()

    print(df_join)

        