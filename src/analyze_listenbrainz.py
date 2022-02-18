#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd
import argparse

from collections import Counter

WIKI_GENRES = "../data/input/wikipedia_EM_genres.csv"
FEAT_DIR = "../data/listenbrainz/feat"
STATS_DIR = "../data/listenbrainz/stats"

def arg_parser():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", type=str, dest='username',
                        help="ListenBrainz Username")
    parser.add_argument("-d", "--date", type=str, dest='date',
                        help="Log Date")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = arg_parser()

    infile = "{}-{}_feat.csv".format(args.username, args.date)
    outfile = infile.replace("feat.csv", "stats.txt")
    infile = os.path.join(FEAT_DIR, infile)
    outfile = os.path.join(STATS_DIR, outfile)

    df = pd.read_csv(infile, delimiter="\t")

    if df.empty:
        raise Exception("No listening logs found!")

    EM_genres = []
    with open(WIKI_GENRES) as inf:
        _reader = csv.reader(inf)
        for row in _reader:
            EM_genres.append(row[0].lower())
    EM_genres.append("electronic")

    genres_list = [x.split(",") for x in df.genres.values if pd.isnull(x) == False]
    genres_list = [y for l in genres_list for y in l]

    genres_found = []
    for genre in EM_genres:
        if genre in genres_list:
            genres_found.append(genre)
        else:
            stringmatch = [x for x in genres_list if genre in x]
            if stringmatch:
                genres_found += stringmatch

    
    percentage = len(genres_found)*100/len(genres_list)
    genres_found = set(genres_found)
    artist_list = [x.split(",") for x in df.artist_name.values if pd.isnull(x) == False]
    artist_list = [y for l in artist_list for y in l]

    isrc_co_list = [x[:2] for x in df.ISRC.values if pd.isnull(x) == False]
    isrc_reg_list = [x[2:5] for x in df.ISRC.values if pd.isnull(x) == False]
    isrc_year_list = [x[5:7] for x in df.ISRC.values if pd.isnull(x) == False]

    file = open(outfile, 'w+')
    file.write("Electronic genres found: {}\n".format(", ".join(genres_found)))
    file.write("Electronic music logs found: {:.2f}% \n\n".format(percentage))
    file.write("Top genres\n")
    for c, el in enumerate(Counter(genres_list).most_common(5)):
        file.write("\t{}: {} ({:.2f}%)\n".format(c+1, el[0], el[1]*100/len(genres_list)))
    file.write("Top artists\n")
    for c, el in enumerate(Counter(artist_list).most_common(5)):
        file.write("\t{}: {} ({:.2f}%)\n".format(c+1, el[0], el[1]*100/len(artist_list)))
    file.write("Top country\n")
    for c, el in enumerate(Counter(isrc_co_list).most_common(5)):
        file.write("\t{}: {} ({:.2f}%)\n".format(c+1, el[0], el[1]*100/len(isrc_co_list)))
    file.write("Top record\n")
    for c, el in enumerate(Counter(isrc_reg_list).most_common(5)):
        file.write("\t{}: {} ({:.2f}%)\n".format(c+1, el[0], el[1]*100/len(isrc_reg_list)))
    file.write("Top year\n")
    for c, el in enumerate(Counter(isrc_year_list).most_common(5)):
        file.write("\t{}: {} ({:.2f}%)\n".format(c+1, el[0], el[1]*100/len(isrc_year_list)))


    file.write("\nFeature statistics\n")
    file.write(df.describe().to_string())
    file.close()
    
    