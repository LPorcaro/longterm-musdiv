#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv


ATT_DIR = "../data/attitudes/"

if __name__ == "__main__":

    att_round = "04"

    ATT_DIR = os.path.join(ATT_DIR, att_round)
    infile = os.path.join(ATT_DIR, 'all.csv')
    outf_g1 = os.path.join(ATT_DIR, 'g1_{}.csv'.format(att_round))
    outf_g2 = os.path.join(ATT_DIR, 'g2_{}.csv'.format(att_round))

    g1, g2 = [], []

    with open("../data/g1_PID.csv", 'r') as inf:
        _reader = csv.reader(inf)
        for row in _reader:
            g1.append(row[0])

    with open("../data/g2_PID.csv", 'r') as inf:
        _reader = csv.reader(inf)
        for row in _reader:
            g2.append(row[0])


    with open(outf_g1, "w+") as outf1, open(outf_g2, "w+") as outf2, open(infile, 'r') as inf:
        _writer1 = csv.writer(outf1, delimiter="\t")
        _writer2 = csv.writer(outf2, delimiter="\t")
        _reader = csv.reader(inf, delimiter="\t")

        header = next(_reader)

        _writer1.writerow(header)
        _writer2.writerow(header)

        for row in _reader:
            if row[23] in g1:
                _writer1.writerow(row)
            elif row[23] in g2:
                _writer2.writerow(row)
