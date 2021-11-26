#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

infile = 'data/track_list_yt_20210923.csv'
df = pd.read_csv(infile, delimiter='\t')

######################################
# ## Genres Stats Write
# outfile = 'Genres_stats.csv'
# with open(outfile, 'w+') as outf:
#     _writer = csv.writer(outf)
#     header = ['genre','count','mean_pop','std_pop','min_pop','q1_pop','q2_pop','q3_pop','max_pop']
#     _writer.writerow(header)
#     for genre in df['genre'].unique():
#         row = df[df['genre'] == genre]['track_pop'].describe().to_csv(index=False, line_terminator=',').replace('track_pop',genre)
#         _writer.writerow(row.split(','))

# # Correlation + scatter plot viewCount - trackPop
# print(df['track_pop'].corr(df['viewCount']))
# plt.figure()
# plt.scatter(df['track_pop'], df['viewCount'])
# plt.ylim(0,1000000000)
# plt.title('Correlation Spotify Track Popularity - YouTube viewCount')
# plt.show()
######################################
# # Plot pop distribution
column = 'track_pop'
nbins = 50
count, mean, std, _min, q1, q2, q3, _max = df[column].describe()
lb, ub = [q1, q3]
df_filt = df[(df[column]>lb) & (df[column] <ub)]
# # Plot without outliers
# df[column].hist(bins=nbins)
# plt.title('Spotify Track Popularity Distribution')
# plt.figure()
# df_filt[column].hist(bins=nbins)
# plt.title('Spotify Track Popularity Distribution Q1-Q3')
# plt.show()
######################################
# Plot viewcount distribution
column = 'viewCount'
count, mean, std, _min, q1, q2, q3, _max = df_filt[column].describe()
lb, ub = [q1, q3]
# df_filt_view = df[(df[column]>lb) & (df[column] <ub)]
df_filt_view = df_filt[(df_filt[column]>lb) & (df_filt[column] <ub)]
# # Plot without outliers
# df_filt[column].hist(bins=2500)
# plt.title('YouTube Viewcount distribution')
# plt.xlim([0,40000000])
# plt.ylim([0,400])
# plt.figure()
# df_filt_view[column].hist(bins=nbins)
# plt.title('YouTube Viewcount distribution Q1-Q3')
# plt.show()
######################################
# ## Genres Stats Filtered Write
# outfile = 'data/genres_stats_filt_20211024.csv'
# with open(outfile, 'w+') as outf:
#     _writer = csv.writer(outf)
#     header = ['genre','count','mean_pop','std_pop','min_pop','q1_pop','q2_pop','q3_pop','max_pop']
#     _writer.writerow(header)
#     for genre in df_filt['genre'].unique():
#         row = df_filt[df_filt['genre'] == genre]['track_pop'].describe().to_csv(index=False, line_terminator=',').replace('track_pop',genre)
#         _writer.writerow(row.split(','))
######################################
# # Get random indexes
n_tracks = 5
idxs = set()
for c, genre in enumerate(df_filt_view['genre'].unique()):
    idx = df_filt_view[df_filt_view['genre'] == genre].index.values
    np.random.shuffle(idx)
    idxs.update(idx[:n_tracks])

df_out = df.filter(items = idxs, axis=0)
df_out.to_csv("data/filtered_tracks_20211124.csv")