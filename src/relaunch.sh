#!/bin/sh

while true; do
  nohup python get_yt_links.py >> test.out
done &