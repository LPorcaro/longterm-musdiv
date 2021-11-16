#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import spotipy
import csv
import os
import time

from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm 

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())


infile = 'data/comm_list_electr_genres.csv'
outfile = 'data/track_list_electr_genres_20210923.csv'

with open(infile, 'r') as inf, open(outfile, 'w+') as outf:
    _reader = csv.reader(inf, delimiter='\t')
    _writer = csv.writer(outf, delimiter='\t')

    HEADER = ['genre', 'pl_id', 't_id', 'artist_name', 'track_name', 
              'track_pop', 'track_isrc']
    _writer.writerow(HEADER)

    for row in _reader:
        genre, pl_id = row

        pl_id = os.path.basename(os.path.normpath(pl_id))

        pl_id = 'spotify:playlist:{}'.format(pl_id)
        offset = 0

        response = sp.playlist_items(pl_id,
                                     offset=offset,
                                     fields='items.track.id,total',
                                     additional_types=['track'])
            

        for t in tqdm((response['items'])):
            tid = t['track']['id']
            urn = 'spotify:track:{}'.format(tid)
            track = sp.track(urn)
            artist_name =  ",".join([a['name'] for a in track['artists']])
            track_name = track['name']
            track_pop = track['popularity']

            try:
                track_isrc = track['external_ids']['isrc']
            except KeyError:
                track_isrc = ''

            row_out = [genre, pl_id, tid, artist_name, track_name, int(track_pop), track_isrc]


            _writer.writerow(row_out)
            time.sleep(0.9)
