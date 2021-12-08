#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
import numpy as np

GENRES_LIST = "data/comm_list_electr_genres.csv"
GENRE_INDEX = "data/genres_index.csv"
GENRE_DIST_MATRIX = "data/genres_distances.npy"

df = pd.read_csv(GENRES_LIST, delimiter="\t")

GenreDict = {x:c for c,x in enumerate(df.genre)}
DistMatrix = np.zeros((len(df.genre), len(df.genre)))

for genre in GenreDict.keys():
    print(genre)

    url = "https://everynoise.com/everynoise1d.cgi?root={}&scope=all".format(genre.replace(" ", "%20"))
    data = requests.get(url).text
    soup = BeautifulSoup(data, 'html.parser')
    tables = soup.find("table")
    for row in tables.find_all('tr'):
        columns = row.find_all('td')
        if columns != []:
            title = columns[0].attrs['title']
            overlap, distance = title.split(',')
            overlap = float(overlap.replace('overlap:', '').strip())
            distance = float(distance.replace('acoustic distance:', '').strip())
            idx = columns[0].text
            g_name = columns[2].text[:-1]

            if any(df.genre == g_name):
                score = 0
                if overlap != 100:
                    score = 0.5*(1-overlap)+0.5*distance
                DistMatrix[GenreDict[genre], GenreDict[g_name]] = score




with open(GENRE_INDEX, "w+") as outf:
    _writer = csv.writer(outf)
    for item in GenreDict.items():
        _writer.writerow(item)

# with open(GENRE_DIST_MATRIX, 'w+') as outf:
#     _writer = csv.writer(outf)
#     for row in DistMatrix:
#         _writer.writerow(row)

np.save(GENRE_DIST_MATRIX, DistMatrix)