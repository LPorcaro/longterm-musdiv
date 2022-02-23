#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import spotipy
import argparse

from pylistenbrainz.errors import ListenBrainzAPIException
from spotipy.oauth2 import SpotifyClientCredentials

JSON_DIR = "../data/listenbrainz/json/{}"
INFO_DIR = "../data/listenbrainz/info/{}"
FEAT_DIR = "../data/listenbrainz/feat/{}"

HEADER_INFO = ["listened_at", "track_name", "artist_name", "isrc", 
               "sp_track_id", "sp_artist_ids"]

HEADER_FEAT = ["sp_track_id", "artist_name", "track_name", "ISRC", 
               "popularity", "genres", "acousticness", "danceability", 
               "instrumentalness", "speechiness", "tempo", "valence", "energy"]


def parse_json(username, date):
    """
    """
    infile = "{}-{}.json".format(username, date)
    outfile_info = infile.replace(".json", ".csv")
    outfile_feat = infile.replace(".json", "_feat.csv")

    if not os.path.exists(JSON_DIR.format(date)):
        os.makedirs(JSON_DIR.format(date))
    infile = os.path.join(JSON_DIR.format(date), infile)

    if not os.path.exists(infile):
        print("User '{}': JSON not found {} ".format(username, infile))
        return

    if not os.path.exists(INFO_DIR.format(date)):
        os.makedirs(INFO_DIR.format(date))   
    outfile_info = os.path.join(INFO_DIR.format(date), outfile_info)

    if not os.path.exists(FEAT_DIR.format(date)):
        os.makedirs(FEAT_DIR.format(date)) 
    outfile_feat = os.path.join(FEAT_DIR.format(date), outfile_feat)

    with open(infile, 'r') as inf:
        listens = json.load(inf)

    tids = []
    # Parse ListenBrainz JSON
    with open(outfile_info, 'w+') as outf:
        _writer = csv.writer(outf, delimiter='\t')
        _writer.writerow(HEADER_INFO)
        for track in listens:
            listened_at = track["listened_at"]
            track_name = track["track_name"]
            artist_name = track["artist_name"]
            isrc = track["isrc"]
            sp_track_id = track["spotify_id"].split("/")[-1]
            sp_artist_ids = ','.join(
                [x.split("/")[-1] for x in 
                 track["additional_info"]["spotify_artist_ids"]])

            row = [listened_at, track_name, artist_name, isrc, 
                   sp_track_id, sp_artist_ids]

            _writer.writerow(row)

            tids.append(sp_track_id)

    # Get info from Spotify
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(),
                         requests_timeout=10,
                         retries=10)
    
    
    with open(outfile_feat, 'w+') as outf:
        _writer = csv.writer(outf, delimiter='\t')
        _writer.writerow(HEADER_FEAT)
        # Get tracks info
        for tids_split in [tids[:50], tids[50:]]:
            if not tids_split:
                continue
            tracks = sp.tracks(tids_split)
            rows = []
            artist_ids = []

            for track, tid in zip(tracks['tracks'], tids_split):
                artist_name = ",".join([a['name'] for a in track['artists']])
                artist_id = [a['id'] for a in track['artists']]
                artist_ids.append(artist_id[0])
                track_name = track['name']
                track_pop = int(track['popularity'])

                try:
                    track_isrc = track['external_ids']['isrc']
                except KeyError:
                    track_isrc = ''


                row_out = [tid, artist_name, track_name, track_isrc, track_pop]
                rows.append(row_out)

            artists = sp.artists(artist_ids)

            rows2 = []
            # Get track features
            tracks_feat = sp.audio_features(tids_split)
            for track_feat in tracks_feat:
                acousticness = float(track_feat['acousticness'])
                danceability = float(track_feat['danceability'])
                instrumentalness = float(track_feat['instrumentalness'])
                speechiness = float(track_feat['speechiness'])
                tempo = float(track_feat['tempo'])
                valence = float(track_feat['valence'])
                energy = float(track_feat['energy'])

                row_out2 = [acousticness, danceability,
                            instrumentalness, speechiness, tempo,
                            valence, energy]

                rows2.append(row_out2)

            for row, row2, art in zip(rows, rows2, artists['artists']):
                new_row = row + [','.join(art['genres'])] + row2 
                _writer.writerow(new_row)

    print("User '{}': JSON parsed {} ".format(username, infile))


def arg_parser():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", type=str, dest='username',
                        help="ListenBrainz Username")
    parser.add_argument("-d", "--date", type=str, dest='date',
                        help="Log Date")
    parser.add_argument("-i", "--input", type=str, dest='input_file',
                        help="Input file with ListenBrainz usernames")
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = arg_parser()

    date = args.date
    username = args.username
    input_file = args.input_file

    if username:
        parse_json(username, date)

    elif input_file:
        infile = open(input_file, 'r')
        lines = infile.readlines()
        infile.close()

        for line in lines:
            try:
                username = line.strip()
                parse_json(username, date)
            except ListenBrainzAPIException:
                print("Problems analyzing logs: {}".format(username))
            
