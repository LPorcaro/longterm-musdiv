from youtubesearchpython import VideosSearch
import csv
from tqdm import tqdm 

outfile = 'track_list_yt_links2.csv'
infile = 'track_list_electr_genres2.csv'
HEADER = ['genre', 'pl_id', 'artist_name', 'track_name', 
              'track_pop', 'track_isrc', 'yt_link']

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
            _writer.writerow([genre, pl_id, artist_name, track_name, track_pop, track_isrc, url])
        except KeyError:
            print (query)
        except IndexError:
            print(query)

        
        
        