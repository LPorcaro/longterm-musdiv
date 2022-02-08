#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pylistenbrainz
import json

from datetime import datetime

now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")

OUT_DIR = "../data/listenbrainz"

if __name__ == "__main__":

    username = 'SingleView'
    client = pylistenbrainz.ListenBrainz()
    listens = client.get_listens(username=username, count=2)
    for listen in listens:
        listen.__dict__['listened_at'] = datetime.fromtimestamp(
            listen.__dict__['listened_at']).isoformat()

    # Serializing json 
    json_object = json.dumps([x.__dict__ for x in listens], indent = 4)
      
    # Writing to sample.json
    outfile = os.path.join(OUT_DIR, "{}-{}.json".format(username, date_time))
    with open(outfile, "w") as outf:
        outf.write(json_object)

