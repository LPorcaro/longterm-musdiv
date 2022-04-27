#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import zipfile

LS_FOLDER = "/home/lorenzo/Downloads/longterm/psytoolkit/LS"
LS_OUT = "../data/ls"


if __name__ == "__main__":
    
    for infile in os.listdir(LS_FOLDER):
        print("Extracting: {}".format(infile))
        outfile = infile.replace(".zip", ".csv")
        infile = os.path.join(LS_FOLDER, infile)
        zf = zipfile.ZipFile(infile) 
        df = pd.read_excel(zf.open('data.xlsx'))

        df_out = df[["PROLIFIC_PID", "familiarity:1", "like:1", "playlist:1"]]

        c_dict = {"familiarity:1":"familiarity",
                  "like:1": "like",
                  "playlist:1": "playlist"}

        df_out = df_out.rename(columns=c_dict)
        df_out = df_out.sort_values(by="PROLIFIC_PID")

        if "G1" in outfile:
            outfile = os.path.join(LS_OUT, 'g1', outfile)
            df_out.to_csv(outfile, index=False)
        elif "G2" in outfile:
            outfile = os.path.join(LS_OUT, 'g2', outfile)
            df_out.to_csv(outfile, index=False)

