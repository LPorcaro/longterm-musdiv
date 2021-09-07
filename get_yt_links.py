from youtubesearchpython import VideosSearch
import csv

outfile = 'track_list_yt_links.csv'

urls = set()

queries = ['NoCopyrightSounds', 'raffaela carra']

for query in queries:
    videosSearch = VideosSearch(query, limit = 1)
    result = videosSearch.result()

    try:
        url = result['result'][0]['link']
    except KeyError:
        print (query)

    urls.add(url)

with open(outfile, 'w+') as outf:
    _writer = csv.writer(outf, delimiter='\t')
    for url in urls:
        _writer.writerow([url])