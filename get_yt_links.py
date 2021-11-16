#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from youtubesearchpython import VideosSearch
import csv
from tqdm import tqdm 

infile = 'data/track_list_electr_genresb.csv'
HEADER = ['genre', 'pl_id', 'artist_name', 'track_name', 
          'track_pop', 'track_isrc', 'yt_link', 'viewCount']

outfile = "data/track_list_yt_20210923b.csv"

with open(outfile, 'w+') as outf, open(infile, 'r') as inf:
    _reader = csv.reader(inf, delimiter='\t')
    _writer = csv.writer(outf, delimiter='\t')
    next(_reader)
    _writer.writerow(HEADER)

    for row in tqdm(_reader):
        genre, pl_id, artist_name, track_name, track_pop, track_isrc = row
        query = ' '.join([artist_name, track_name]).replace("'", ' ')
        videosSearch = VideosSearch(query, limit = 1)
        result = videosSearch.result()


        try:
            url = result['result'][0]['link']
            viewCount = result['result'][0]['viewCount']['text'].replace('views', '').replace('view','').replace(',', '').strip()
            if viewCount == 'No':
                viewCount = 0
            else:
                viewCount = int(viewCount)

            _writer.writerow([genre, pl_id, artist_name, track_name, track_pop, track_isrc, url, viewCount])
        except KeyError:
            print (query)
        except IndexError:
            print(query)

        
        
        