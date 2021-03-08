#/bin/bash

rm -f samples/inputs/*
FPS=$(ffprobe input.mp4 2>&1 | grep fps | sed 's/^.*, \(.*\) fps.*$/\1/')
ffmpeg -i input.mp4 -q:v 5 samples/inputs/img%03d.jpg
python test.py
ffmpeg -i samples/results/img%03d.jpg -r $FPS output.mp4
