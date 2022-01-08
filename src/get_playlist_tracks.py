#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import spotipy
import csv
import os
import time

from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm


infile = '../data/input/spotify_EM_playlists2.csv'
outfile = '../data/input/tracklist_20220104_2.csv'
outfile_feat = '../data/input/tracklist_features_20220104_2.csv'

HEADER = ['genre', 'maingenre', 'sp_id', 'artist_name', 'track_name',
          'ISRC', 'pop']

HEADER2 = ['sp_id', 'acousticness', 'danceability', 'instrumentalness',
           'speechiness', 'tempo', 'valence', 'energy']


if __name__ == "__main__":

    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(),
                         requests_timeout=10,
                         retries=10)

    with open(infile, 'r') as inf, open(outfile, 'w+') as outf, \
         open(outfile_feat, 'w+') as outf2:

        _reader = csv.reader(inf)
        _writer = csv.writer(outf, delimiter='\t')
        _writer2 = csv.writer(outf2, delimiter='\t')

        _writer.writerow(HEADER)
        _writer2.writerow(HEADER2)

        next(_reader)

        for row in tqdm(_reader):
            genre, maingenre, pl_id = row
            pl_id = os.path.basename(os.path.normpath(pl_id))
            pl_id = 'spotify:playlist:{}'.format(pl_id)
            offset = 0

            response = sp.playlist_items(pl_id,
                                         offset=offset,
                                         fields='items.track.id,total',
                                         additional_types=['track'])

            # Get track ids
            tids = []
            for t in response['items']:
                tid = t['track']['id']
                tids.append(tid)

            # Get tracks info
            for s, tids_split in enumerate([tids[:50], tids[50:]]):

                tracks = sp.tracks(tids_split)
                for c, track in enumerate(tracks['tracks']):
                    artist_name = ",".join([a['name'] for a in track['artists']])
                    track_name = track['name']
                    track_pop = int(track['popularity'])

                    try:
                        track_isrc = track['external_ids']['isrc']
                    except KeyError:
                        track_isrc = ''

                    if s == 1:
                        c += 50

                    row_out = [genre, maingenre, tids[c], artist_name,
                               track_name, track_isrc, track_pop]
                    _writer.writerow(row_out)
                    

            # Get track features
            tracks_feat = sp.audio_features(tids)
            for c, track_feat in enumerate(tracks_feat):
                acousticness = float(track_feat['acousticness'])
                danceability = float(track_feat['danceability'])
                instrumentalness = float(track_feat['instrumentalness'])
                speechiness = float(track_feat['speechiness'])
                tempo = float(track_feat['tempo'])
                valence = float(track_feat['valence'])
                energy = float(track_feat['energy'])

                row_out2 = [tids[c], acousticness, danceability,
                            instrumentalness, speechiness, tempo,
                            valence, energy]

                _writer2.writerow(row_out2)

            time.sleep(0.8)