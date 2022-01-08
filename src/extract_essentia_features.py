#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import pandas as pd

from tqdm import tqdm 

AUDIO_DIR = "/home/lorenzo/Data/longterm_data/audios_new"
FEAT_DIR = "/home/lorenzo/Data/longterm_data/features"
TRACKS = "../data/input/random_tracklist_20220104.csv"

PROFILE = "/home/lorenzo/Workspace/essentia/essentia-extractors-v2.1_beta2/test_profile.yaml"

if __name__ == "__main__":


    df_tracks = pd.read_csv(TRACKS, delimiter='\t')

    for yt_id in tqdm(df_tracks.yt_id.values):
        filename = yt_id + '.mp3'
        audiofile = os.path.join(AUDIO_DIR, filename)
        outfilename = yt_id + '.json'
        jsonfile = os.path.join(FEAT_DIR, outfilename)

        if os.path.exists(jsonfile):
            print("Feature file found: {}".format(jsonfile))
            continue

        if os.path.exists(audiofile):
            print("Extracting features file: {}".format(audiofile))
            subprocess.call(["essentia_streaming_extractor_music",
                             "{}".format(audiofile),
                             "{}".format(jsonfile),
                             "{}".format(PROFILE)])

        else:
            print("Audio not found: {}".format(audiofile))

