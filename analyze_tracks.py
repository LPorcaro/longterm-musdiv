import pandas as pd
import csv

infile = 'TrackListYT.csv'
outfile = 'Genres_stats.csv'

df = pd.read_csv(infile)

with open(outfile, 'w+') as outf:
    _writer = csv.writer(outf)
    header = ['genre','count','mean_pop','std_pop','min_pop','q1_pop','q2_pop','q3_pop','max_pop']
    _writer.writerow(header)
    for genre in df['genre'].unique():
        row = df[df['genre'] == genre]['track_pop'].describe().to_csv(index=False, line_terminator=',').replace('track_pop',genre)
        _writer.writerow(row.split(','))

        