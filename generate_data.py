#!/usr/bin/env python3

# Original files are generated with
# ffmpeg -i input_file -q:v 5 original/img%06d.jpg

import os
import random

from PIL import Image

num_clips = 250
num_rows_per_clip = 2
num_cols_per_clip = 4
num_windows_per_clip = num_rows_per_clip * num_cols_per_clip
num_frames_per_clip = 30
window_size = 256
image_scale = 1024 / 1920

original_path = 'original'
original_file_format = original_path + '/img{:06d}.jpg'

output_path_format = 'dataset/clip_window_{:04d}'
output_file_format = output_path_format + '/frame_{:02d}.jpg'

def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    num_files = len(os.listdir(original_path))

    clip_starting_points = list(range(1, num_files - num_frames_per_clip + 1, num_frames_per_clip))
    random.shuffle(clip_starting_points)
    clip_starting_points = clip_starting_points[:num_clips]

    for clip, starting_point in enumerate(clip_starting_points):
        print(f'Processing clip ({clip + 1}/{num_clips})...', end='\r')

        for clip_window in range(num_clips * num_windows_per_clip):
            ensure_path(output_path_format.format(clip_window))

        for frame in range(num_frames_per_clip):
            im = Image.open(original_file_format.format(starting_point + frame))
            width, height = im.size
            width = int(width * image_scale)
            height = int(height * image_scale)
            im = im.resize((width, height))
            starting_height = (height - num_rows_per_clip * window_size) // 2
            starting_width = (width - num_cols_per_clip * window_size) // 2

            for row in range(num_rows_per_clip):
                for col in range(num_cols_per_clip):
                    window = row * num_cols_per_clip + col
                    cropped = im.crop((
                        starting_width + col * window_size,
                        starting_height + row * window_size,
                        starting_width + (col + 1) * window_size,
                        starting_height + (row + 1) * window_size
                    ))
                    cropped.save(output_file_format.format(clip * num_windows_per_clip + window, frame))

if __name__ == '__main__':
    main()
