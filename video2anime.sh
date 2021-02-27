#/bin/bash

rm -f samples/inputs/*
ffmpeg -i input.mp4 -q:v 5 samples/inputs/img%03d.jpg
python test.py --device cpu
ffmpeg -i samples/results/img%03d.jpg -r 30 output.mp4
