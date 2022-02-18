#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pylistenbrainz
import json
import argparse

from datetime import datetime
from tqdm import tqdm 

now = datetime.now()
date_time = now.strftime("%Y%m%d")

OUT_DIR = "../data/listenbrainz/json"


def arg_parser():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", type=str, dest='username',
                        help="ListenBrainz Username")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = arg_parser()

    username = args.username
    client = pylistenbrainz.ListenBrainz()

    tot_listes = client.get_user_listen_count(username)
    if tot_listes > 100:
        tot_listes = 100

    listens = client.get_listens(username=username, count=tot_listes)
    for listen in tqdm(listens):
        listen.__dict__['listened_at'] = datetime.fromtimestamp(
            listen.__dict__['listened_at']).isoformat()

    # Serializing json 
    json_object = json.dumps([x.__dict__ for x in listens], indent = 4)
      
    # Writing to sample.json
    outfile = os.path.join(OUT_DIR, "{}-{}.json".format(username, date_time))
    with open(outfile, "w") as outf:
        outf.write(json_object)

    print("Found {} listen events".format(tot_listes))