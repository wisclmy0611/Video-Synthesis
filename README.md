# Video-Synthesis

## Dataset Generation

1. Prepare your images (which should be named `imgXXXXXX.jpg`) in a directory called `original`.
2. Run `./generate_data.py`. It will make a `dataset` directory containing generated data.

## Baseline Generation

1. Clone the [animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch) repository and install the dependencies specified in it.
2. Install [`ffmpeg`](https://ffmpeg.org/).
3. Copy [`video2anime.sh`](video2anime.sh) to your cloned repository.
4. Prepare an `input.mp4` as your input video inside the cloned repository.
5. Run `./video2anime.sh`. The script will automatically convert your video to images, run all of them through the model, and convert the resulting images back to a video called `output.mp4`.

## Reference

- [ffmpeg](https://ffmpeg.org/)
- [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)
- [animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)
