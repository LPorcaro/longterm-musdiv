#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import zipfile
import csv

LS_FOLDER = "/home/lorenzoporcaro/Downloads/longterm/LS"
LS_OUT = "../data/ls"


if __name__ == "__main__":

    HD = []
    with open("../data/HD_PID.csv", 'r') as inf:
        _reader = csv.reader(inf)
        for row in _reader:
            HD.append(row[0])
    LD = []
    with open("../data/LD_PID.csv", 'r') as inf:
        _reader = csv.reader(inf)
        for row in _reader:
            LD.append(row[0])

    
    for infile in sorted(os.listdir(LS_FOLDER)):
        print("Extracting: {}".format(infile))
        outfile = infile.replace(".zip", ".csv")
        outfile = outfile.replace("G1", "HD")
        outfile = outfile.replace("G2", "LD")
        infile = os.path.join(LS_FOLDER, infile)
        zf = zipfile.ZipFile(infile) 
        df = pd.read_excel(zf.open('data.xlsx'))

        df_out = df[["PROLIFIC_PID", "familiarity:1", "like:1", "playlist:1"]]

        c_dict = {"familiarity:1":"familiarity",
                  "like:1": "like",
                  "playlist:1": "playlist"}

        df_out = df_out.rename(columns=c_dict)
        df_out = df_out.sort_values(by="PROLIFIC_PID")


        if "HD" in outfile:
            if df['PROLIFIC_PID'].isnull().values.any():
                print(set(HD).difference(set(df['PROLIFIC_PID'].tolist())))
            df_out = df_out[df_out.PROLIFIC_PID.isin(HD)]
            df_out = df_out.drop_duplicates()
            outfile = os.path.join(LS_OUT, 'HD', outfile)
            df_out.to_csv(outfile, index=False)
        elif "LD" in outfile:
            if df['PROLIFIC_PID'].isnull().values.any():
                print(set(LD).difference(set(df['PROLIFIC_PID'].tolist())))
            df_out = df_out[df_out.PROLIFIC_PID.isin(LD)]
            df_out = df_out.drop_duplicates()
            outfile = os.path.join(LS_OUT, 'LD', outfile)
            df_out.to_csv(outfile, index=False)

