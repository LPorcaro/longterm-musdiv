#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

EMB_DIR = "../data/embeddings/{}"


def import_embeddings(emb_dir):
    """
    """
    embeddings = []
    filenames = []
    emb_dir_input = emb_dir.format('musicnn')
    for emb_file in os.listdir(emb_dir_input):
        if not emb_file.endswith('.npy'):
            continue

        filename, ext = os.path.splitext(emb_file)

        file_path = os.path.join(emb_dir_input, emb_file)
        if os.path.exists(file_path):
            embedding = np.load(file_path)
            if len(embedding) != 200:
                print(filename)
            else:
                embeddings.append(embedding)
                filenames.append(filename)

    return embeddings, filenames


def reduce_embeddings(embeddings, filenames):
    """
    """
    # PCA
    embeddings_stacked = np.vstack(embeddings)
    projection = PCA(random_state=0, copy=False)
    projection = projection.fit(embeddings_stacked[:, :None])

    threshold = 0.8
    pc_num = 1
    exp_var_ratio = 0

    while exp_var_ratio <= threshold:
        exp_var_ratio = np.sum(projection.explained_variance_ratio_[:pc_num])
        pc_num += 1

    print("[PCA] Explained variance ratio by {} PC: {}".
          format(pc_num, exp_var_ratio))

    projection = PCA(random_state=0, copy=False, n_components=pc_num)
    embeddings_reduced = projection.fit_transform(embeddings_stacked[:, :None])

    for c, emb in enumerate(embeddings_reduced):
        outfile = os.path.join(EMB_DIR.format('musicnn_pca'), filenames[c])
        np.save(outfile, emb)

    # TSNE
    projection = TSNE(n_components=2, perplexity=7, random_state=1,
                      n_iter=500, init='pca', verbose=True)
    embeddings_reduced = projection.fit_transform(embeddings_reduced[:, :None])

    for c, emb in enumerate(embeddings_reduced):
        outfile = os.path.join(EMB_DIR.format('musicnn_tsne'), filenames[c])
        np.save(outfile, emb)

    return embeddings_reduced


if __name__ == "__main__":
    """
    """


    embeddings, filenames = import_embeddings(EMB_DIR)

    embeddings_reduced = reduce_embeddings(embeddings, filenames)
