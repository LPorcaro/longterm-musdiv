#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

infile = '../data/input/tracklist_yt_20220104.csv'
outfile = '../data/input/random_tracklist_20220104.csv'
outfile_genre = '../data/input/random_tracklist_genrestats_20220104.csv'


def plot_popularity(df):
    """
    """
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3)

    nbins = 25

    # Plot without outliers
    df['pop'].hist(bins=nbins, ax=ax1)
    ax1.set_title('Spotify Track Popularity Distribution')

    # Plot without outliers
    df['viewCount'].hist(bins=nbins, ax=ax2)
    ax2.set_title('YouTube viewCount Distribution')

    # Correlation + scatter plot viewCount - trackPop
    ax3.scatter(df['pop'], df['viewCount'])
    ax3.set_title(r'Spotify Popularity - YouTube viewCount ($\rho={:.2f}$)'.
        format(df['pop'].corr(df['viewCount'])))

    plt.show()


def write_genre_stats(df):
    """
    """
    with open(outfile_genre, 'w+') as outf:
        _writer = csv.writer(outf)
        header = ['maingenre', 'genre', 'count', 'mean',
                  'std', 'min', 'q1', 'q2', 'q3', 'max']

        _writer.writerow(header)
        for genre in df['genre'].unique():
            row = df[df['genre'] == genre]['pop'].describe().to_csv(
                index=False, sep='\t', line_terminator=',').replace(
                'pop', genre)

            _writer.writerow(
                df[df.genre == genre].maingenre.unique().tolist() +
                row.split(','))


if __name__ == "__main__":

    df = pd.read_csv(infile, delimiter='\t')

    # Filter viewCount oustide [Q1, Q3]
    column = 'viewCount'
    count, mean, std, _min, q1, q2, q3, _max = df[column].describe()
    df_filt = df[(df[column] > q1) & (df[column] < q3)]

    # Get random tracks
    n_tracks = 10
    idxs = set()
    for c, genre in enumerate(df_filt['genre'].unique()):
        idx = df_filt[df_filt['genre'] == genre].index.values
        np.random.shuffle(idx)
        idxs.update(idx[:n_tracks])

    df_out = df.filter(items=idxs, axis=0)
    df_out = df_out.drop_duplicates(subset=('sp_id'), keep='last')
    df_out.to_csv(outfile, sep='\t', index=False)


    plot_popularity(df_out)
    write_genre_stats(df_out)
