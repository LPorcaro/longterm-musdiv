#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import pandas as pd

from http.client import RemoteDisconnected
from youtubesearchpython import VideosSearch
from tqdm import tqdm

infile = '../data/input/tracklist_20220104.csv'
outfile = '../data/input/tracklist_yt_20220104.csv'

HEADER = ['genre', 'maingenre', 'sp_id', 'yt_id', 'artist_name', 'track_name',
          'ISRC', 'pop', 'viewCount', 'yt_link']


if __name__ == "__main__":

    df_in = pd.read_csv(infile, delimiter='\t')
    # Get tracks with popularity outside [Q1, Q3]
    count, mean, std, _min, q1, q2, q3, _max = df_in['pop'].describe()
    df_filt = df_in[(df_in['pop'] > q1) & (df_in['pop'] < q3)]

    df_out = pd.read_csv(outfile, delimiter='\t')

    with open(outfile, 'a') as outf, open(infile, 'r') as inf:
        _reader = csv.reader(inf, delimiter='\t')
        _writer = csv.writer(outf, delimiter='\t')
        # _writer.writerow(HEADER)

        next(_reader)
        for row in tqdm(_reader):
            [genre, maingenre, tid, artist_name, track_name,
             track_isrc, track_pop] = row

            # Filter by popularity
            if tid not in df_filt.sp_id.values:
                continue
            # Skip if already found
            elif tid in df_out.sp_id.values:
                continue

            try:
                query = ' '.join([artist_name, track_name]).replace("'", ' ')
                videosSearch = VideosSearch(query, limit=1)
                result = videosSearch.result()

                url = result['result'][0]['link']
                viewCount = result['result'][0]['viewCount']['text']
                viewCount = viewCount.replace('views', '').replace('view', '')
                viewCount = viewCount.replace(',', '').strip()

                if viewCount == 'No':
                    viewCount = 0
                else:
                    viewCount = int(viewCount)

                yt_id = url.split('?v=')[1]
                print(yt_id)
                row = [genre, maingenre, tid, yt_id, artist_name, track_name,
                       track_isrc, track_pop, viewCount, url]

                _writer.writerow(row)

            except KeyError:
                print(query)
            except IndexError:
                print(query)
            except RemoteDisconnected:
                print(query)
