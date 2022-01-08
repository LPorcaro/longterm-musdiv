#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import essentia.standard as es
import numpy as np
import subprocess
import pandas as pd

from tqdm import tqdm
from statistics import mean


MUSICNN = "/home/lorenzo/Workspace/music-explore/essentia-tf-models/msd-musicnn.pb"
AUDIO_FOLDER = "/home/lorenzo/Data/longterm_data/audios_new"
MUSICNN_FOLDER = "../data/embeddings/musicnn"
TRACKS = "../data/input/random_tracklist_20220104.csv"


if __name__ == "__main__":

    df_tracks = pd.read_csv(TRACKS, delimiter='\t')

    for yt_id in tqdm(df_tracks.yt_id.values):

        emb_file = os.path.join(MUSICNN_FOLDER, yt_id + '.npy' )
        audiofile = os.path.join(AUDIO_FOLDER, yt_id + '.mp3')

        if os.path.exists(emb_file):
            print("Embedding already extracted: {}".format(emb_file))
            continue
        elif not os.path.exists(audiofile):
            print("Audio not found: {}".format(audiofile))
        else:
            print("Extracting: {}".format(audiofile))

            audiofile = os.path.join(AUDIO_FOLDER, audiofile)
            
            audio = es.MonoLoader(filename="{}".format(audiofile), sampleRate=16000)()

            musicnn_embs = es.TensorflowPredictMusiCNN(graphFilename=MUSICNN,
                                                       output='model/dense/BiasAdd')(audio)

            embedding = list(map(mean, zip(*musicnn_embs)))

            np.save(emb_file, embedding)
