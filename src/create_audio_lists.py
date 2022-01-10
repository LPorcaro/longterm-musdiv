#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import csv
import numpy as np
import os
import sox
import random

from itertools import combinations
from operator import itemgetter


marker_types = [".", "o", "v", "^", "<",
                ">", "1", "2", "3", "4",
                "8", "s", "p", "P", "h",
                "H", "+", "x", "X", "D",
                ".", "o", "v", "^", "<", '1']

np.random.shuffle(marker_types)


AUDIO_FOLDER = "/home/lorenzo/Data/longterm_data/audios_new"
AUDIO_FOLDER_OUT = "/home/lorenzo/Data/longterm_data/audios_trim/list_{}"
AUDIO_LISTS_OUT = "/home/lorenzo/Data/longterm_data/audios_list"


TRACKS = "../data/input/random_tracklist_20220104.csv"

CREATION_TIME = "20220110_112852"
LIST_DIV = "../data/lists/track_list_div_{}.csv".format(CREATION_TIME)
LIST_NOT_DIV = "../data/lists/track_list_not_div_{}.csv".format(CREATION_TIME)


def import_lists():
    """
    """

    list_div, list_not_div = [], []
    list_div_genres, list_not_div_genres = [], []

    with open(LIST_DIV, 'r') as inf1, open(LIST_NOT_DIV, 'r') as inf2:
        _reader1 = csv.reader(inf1)
        _reader2 = csv.reader(inf2)

        for row in _reader1:
            list_div.append(row[:-1])
            list_div_genres.append(row[-1])

        for row in _reader2:
            list_not_div.append(row[:-1])
            list_not_div_genres.append(row[-1])

    return list_div, list_not_div, list_div_genres, list_not_div_genres


if __name__ == "__main__":

    (list_div, list_not_div,
        list_div_genres, list_not_div_genres) = import_lists()


    for c, tracklist in enumerate(list_not_div):
        audioslist_file = []
        for yt_id in tracklist:
            audiofile = yt_id + '.mp3'
            file_path = os.path.join(AUDIO_FOLDER, audiofile)
            file_out_path = os.path.join(AUDIO_FOLDER_OUT.format(c), audiofile)
            if not os.path.exists(AUDIO_FOLDER_OUT.format(c)):
                os.makedirs(AUDIO_FOLDER_OUT.format(c))
            if os.path.exists(file_path):
                print("Trimming file {}".format(file_path))
                sample_rate = sox.file_info.sample_rate(file_path)
                dur = sox.file_info.duration(file_path)
                start = random.randrange(int(dur)-70)
                # create transformer
                tfm = sox.Transformer()
                # convert sample rate
                if sample_rate != 48000.0:
                    tfm.convert(samplerate=48000.0)
                # trim the audio between 5 and 10.5 seconds.
                tfm.trim(start, start + 60)
                # apply norm
                tfm.norm()
                # apply a fade in and fade out
                tfm.fade(fade_in_len=1.0, fade_out_len=1.0)
                # create an output file.
                tfm.build_file(file_path, file_out_path)
                print("Created file {}".format(file_out_path))
                audioslist_file.append(file_out_path)

        # create combiner
        cbn = sox.Combiner()
        # convert output to 8000 Hz stereo
        cbn.convert(samplerate=48000, n_channels=2)
        # create the output file
        outfile = os.path.join(AUDIO_LISTS_OUT, 'List{}.mp3'.format(c))
        cbn.build(audioslist_file, outfile, 'concatenate')
        print("Created file {}".format(outfile))


    