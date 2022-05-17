#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse


ROUNDS =  ["00", "01", "02", "03", "04", "10"]


def extract(att_round):
    """
    """
    ATT_DIR = "../data/attitudes/"
    ATT_DIR = os.path.join(ATT_DIR, att_round)
    infile = os.path.join(ATT_DIR, 'all.csv')
    outf_g1 = os.path.join(ATT_DIR, 'HD_{}.csv'.format(att_round))
    outf_g2 = os.path.join(ATT_DIR, 'LD_{}.csv'.format(att_round))

    HD, LD = [], []

    with open("../data/HD_PID.csv", 'r') as inf:
        _reader = csv.reader(inf)
        for row in _reader:
            HD.append(row[0])

    with open("../data/LD_PID.csv", 'r') as inf:
        _reader = csv.reader(inf)
        for row in _reader:
            LD.append(row[0])


    with open(outf_g1, "w+") as outf1, open(outf_g2, "w+") as outf2, open(infile, 'r') as inf:
        _writer1 = csv.writer(outf1, delimiter="\t")
        _writer2 = csv.writer(outf2, delimiter="\t")
        _reader = csv.reader(inf, delimiter="\t")

        header = next(_reader)

        _writer1.writerow(header)
        _writer2.writerow(header)

        for row in _reader:
            if row[23] in HD:
                _writer1.writerow(row)
            elif row[23] in LD:
                _writer2.writerow(row)   


def arg_parser():
    """
    """
    parser = argparse.ArgumentParser()
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
        extract(att_round)
    else:
        for att_round in ROUNDS:
            extract(att_round)