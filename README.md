# Video-Synthesis

## Dataset Generation

1. Prepare your images (which should be named `imgXXXXXX.jpg`) in a directory called `original`.
2. Run `./generate_data.py`. It will make a `dataset` directory containing generated data.

## Baseline Generation

1. Install [`ffmpeg`](https://ffmpeg.org/).
2. Prepare an `input.mp4` as your input video.
3. Run `./video2anime.sh`. The script will automatically convert your video to images, run all of them through the model, and convert the resulting images back to a video called `output.mp4`.

## Reference

- [ffmpeg](https://ffmpeg.org/)
- [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)
- [animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)
- [cartoonGAN-pytorch](https://github.com/znxlwm/pytorch-CartoonGAN)
