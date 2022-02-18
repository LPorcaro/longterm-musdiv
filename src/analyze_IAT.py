#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd


if __name__ == "__main__":

    INPUT_FOLDER = "../data/iat"
    input_file = "iat.2022-02-18-0140.data.fc1ef2b5-0515-477f-8053-6e1ce0357cdf.txt"
    input_file = os.path.join(INPUT_FOLDER, input_file)

    df = pd.DataFrame(columns = ["block", "block_num", "item", "rt", "status"])

    with open(input_file, 'r') as inf:
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
            # print(row)

    pen_left = round(df[df.block == "mix_compatible"].rt.mean() + 400)
    pen_right = round(df[df.block == "mix_incompatible"].rt.mean() + 400)

    df.loc[(df.status == 2) & (df.block == "mix_compatible"), 'rt'] = pen_left
    df.loc[(df.status == 2) & (df.block == "mix_incompatible"), 'rt'] = pen_right

    avg_left = df[df.block == "mix_compatible"].rt.mean()
    avg_right = df[df.block == "mix_incompatible"].rt.mean()
    all_std = df[df.status == 1].rt.std()

    dscore = (avg_left - avg_right) / all_std
    print(dscore)

