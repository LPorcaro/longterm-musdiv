#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pylistenbrainz
import json
import argparse

from pylistenbrainz.errors import ListenBrainzAPIException
from datetime import datetime
from tqdm import tqdm
from operator import itemgetter


def get_listen_logs(username, out_dir):
    """
    """
    date_time = datetime.now().strftime("%Y%m%d")
    user_folder = os.path.join(out_dir, username)
    if os.path.exists(user_folder):
        for json_file in sorted(os.listdir(user_folder), reverse=True):
            infile = os.path.join(user_folder, json_file)
            with open(infile, 'r') as inf:
                listens = json.load(inf)
            last_listen = listens[0]
            last_listen_date = last_listen['listened_at']
            last_listen_obj = datetime.fromisoformat(last_listen_date)
            break

        listens = []
        listens_batch_flag = 1
        min_ts = (int(last_listen_obj.timestamp()))

        while listens_batch_flag:
            listens_batch = client.get_listens(username=username,
                                               min_ts=min_ts,
                                               count=100)

            if len(listens_batch) == 0:
                listens_batch_flag = 0
                break

            min_ts = listens_batch[0].__dict__['listened_at']

            for listen in tqdm(listens_batch):
                listen.__dict__['listened_at'] = datetime.fromtimestamp(
                    listen.__dict__['listened_at']).isoformat()

            listens.extend(listens_batch)

    else:
        os.makedirs(user_folder)

        listens = []
        listens_count = client.get_user_listen_count(username)
        max_ts = int(datetime.now().timestamp())

        while len(listens) < listens_count:
            listens_batch = client.get_listens(username=username,
                                               max_ts=max_ts,
                                               count=100)

            max_ts = listens_batch[-1].__dict__['listened_at']

            for listen in tqdm(listens_batch):
                listen.__dict__['listened_at'] = datetime.fromtimestamp(
                    listen.__dict__['listened_at']).isoformat()

            listens.extend(listens_batch)

    listens = sorted([x.__dict__ for x in listens],
                     key=itemgetter('listened_at'),
                     reverse=True)

    # Serializing json
    json_object = json.dumps([x for x in listens], indent=4)

    # Writing to sample.json
    out_dir = os.path.join(out_dir, username)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    outfile = os.path.join(out_dir, "{}-{}.json".format(username, date_time))
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
    parser.add_argument("-o", "--output", type=str, dest='out_dir',
                        help="Output for JSON files")
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = arg_parser()
    username = args.username
    input_file = args.input_file
    out_dir = args.out_dir

    if not username and not input_file:
        raise ValueError("Input not specified")
    elif not out_dir:
        raise ValueError("Output folder not specified")

    client = pylistenbrainz.ListenBrainz()

    if username:
        get_listen_logs(username, out_dir)
    elif input_file:
        infile = open(input_file, 'r')
        lines = infile.readlines()
        infile.close()
        for line in lines:
            try:
                username = line.strip()
                get_listen_logs(username, out_dir)
            except ListenBrainzAPIException:
                print("Problems retrieving logs: {}".format(username))
