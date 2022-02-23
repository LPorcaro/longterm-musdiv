#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pylistenbrainz
import json
import argparse

from pylistenbrainz.errors import ListenBrainzAPIException
from datetime import datetime, timedelta
from tqdm import tqdm 

max_ts = datetime.now()
min_ts = max_ts - timedelta(hours=24)

date_time = max_ts.strftime("%Y%m%d")

OUT_DIR = "../data/listenbrainz/json/{}".format(date_time)


def get_listen_logs(username):
    """
    """
    tot_listes = client.get_user_listen_count(username)
    if tot_listes > 100:
        tot_listes = 100

    listens = client.get_listens(username=username, 
                                 min_ts=int(min_ts.timestamp()),
                                 max_ts = int(max_ts.timestamp()))

    for listen in tqdm(listens):
        listen.__dict__['listened_at'] = datetime.fromtimestamp(
            listen.__dict__['listened_at']).isoformat()

    # Serializing json 
    json_object = json.dumps([x.__dict__ for x in listens], indent=4)
      
    # Writing to sample.json
    outfile = os.path.join(OUT_DIR, "{}-{}.json".format(username, date_time))
    with open(outfile, "w") as outf:
        outf.write(json_object)

    print("User '{}': Found {} listen events".format(username, len(listens)))    


def arg_parser():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", type=str, dest='username',
                        help="ListenBrainz Username")
    parser.add_argument("-i", "--input", type=str, dest='input_file',
                        help="Input file with ListenBrainz usernames")
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = arg_parser()

    username = args.username

    input_file = args.input_file

    client = pylistenbrainz.ListenBrainz()

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if username:
        get_listen_logs(username)
    elif input_file:
        infile = open(input_file, 'r')
        lines = infile.readlines()
        infile.close()

        for line in lines:
            try:
                username = line.strip()
                get_listen_logs(username)
            except ListenBrainzAPIException:
                print("Problems retrieving logs: {}".format(username))
