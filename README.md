# Video-Synthesis

## Dataset Generation

1. Prepare your images (which should be named `imgXXXXXX.jpg`) in a directory called `original`.
2. Run `./generate_data.py`. It will make a `dataset` directory containing generated data.

## Baseline Generation

1. Clone the [animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch) repository and install the dependencies specified in it.
2. Install [`ffmpeg`](https://ffmpeg.org/).
3. Copy [`video2anime.sh`](video2anime.sh) and [`pytorch_generator_Paprika.pt`](pytorch_generator_Paprika.pt) to your cloned repository.
4. Prepare an `input.mp4` as your input video inside the cloned repository.
5. Run `./video2anime.sh`. The script will automatically convert your video to images, run all of them through the model, and convert the resulting images back to a video called `output.mp4`.

## Training

Run `python train.py` to train the model. You can also use the following flags to tweak the training:
    - `--dataset`: which dataset to use (Default: `dataset`).
    - `--batch_size`: the batch size (Default: `64`).
    - `--device`: device to train (Default: `cuda:0`).
    - `--train_epoch`: number of training epoch (Default: `100`).
    - `--lrD`: discriminator learning rate (Default: `4e-5`).
    - `--lrG`: generator learning rate (Default: `2e-5`).

## Evaluation

Run `python test.py` for inference.

## Reference

- [ffmpeg](https://ffmpeg.org/)
- [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)
- [animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)
- [cartoonGAN-pytorch](https://github.com/znxlwm/pytorch-CartoonGAN)
