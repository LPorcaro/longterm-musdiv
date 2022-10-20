#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import subprocess
import pandas as pd

from essentia.standard import MonoLoader, TensorflowPredictVGGish, TensorflowPredictMusiCNN
from tqdm import tqdm
from statistics import mean

VGGISH = "essentia_extractors/  "
MUSICNN = "/home/lorenzo/Workspace/music-explore/essentia-tf-models/msd-musicnn.pb"
MTT_MUSICNN = "essentia_extractors/mtt-musicnn-1.pb"

MUSICNN_FOLDER = "../data/embeddings/musicnn"
VGGISH_FOLDER = "../data/embeddings/vggish"
MTT_MUSICNN_FOLDER = "../data/embeddings/mtt_musicnn"

AUDIO_FOLDER = "/home/lorenzoporcaro/Data/longterm-data/audios_new"
TRACKS = "../data/input/random_tracklist_20220104.csv"


if __name__ == "__main__":

    df_tracks = pd.read_csv(TRACKS, delimiter='\t')

    for yt_id in tqdm(df_tracks.yt_id.values):

        emb_file = os.path.join(MTT_MUSICNN_FOLDER, yt_id + '.npy' )
        audiofile = os.path.join(AUDIO_FOLDER, yt_id + '.mp3')

        if os.path.exists(emb_file):
            print("Embedding already extracted: {}".format(emb_file))
            continue
        elif not os.path.exists(audiofile):
            print("Audio not found: {}".format(audiofile))
        else:
            print("Extracting: {}".format(audiofile))

            audiofile = os.path.join(AUDIO_FOLDER, audiofile)
            
            audio = MonoLoader(filename="{}".format(audiofile), sampleRate=16000)()

            # musicnn_embs = TensorflowPredictMusiCNN(graphFilename=MTT_MUSICNN,
            #                                            output='model/dense/BiasAdd')(audio)

            vggish_embs = TensorflowPredictVGGish(graphFilename=VGGISH,
                                                  input="melspectrogram",
                                                  output="embeddings")(audio)

            embedding = list(map(mean, zip(*musicnn_embs)))

            # np.save(emb_file, embedding)
