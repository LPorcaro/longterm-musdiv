#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import pandas as pd

from tqdm import tqdm 

AUDIO_DIR_OLD = "/home/lorenzo/Data/longterm_data/audios"
AUDIO_DIR_NEW = "/home/lorenzo/Data/longterm_data/audios_new"

TRACKS = "../data/input/random_tracklist_20220104.csv"


if __name__ == "__main__":


    df_tracks = pd.read_csv(TRACKS, delimiter='\t')

    for yt_id in tqdm(df_tracks.yt_id.values):
        filename = yt_id + '.mp3'
        file_path_new = os.path.join(AUDIO_DIR_NEW, filename)

        if os.path.exists(file_path_new):
            print("Found file: {}".format(file_path_new))
        else:
            print("Downloading file: {}".format(file_path_new))
            file_url = "https://www.youtube.com/watch?v={}".format(yt_id)
            subprocess.call(["yt-dlp", 
                             "--output", 
                             "{}/%(id)s.%(ext)s".format(AUDIO_DIR_NEW), 
                             "-x" , 
                             "--audio-format", 
                             "mp3", 
                             "-i", 
                             "{}".format(file_url)])
            

