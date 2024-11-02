# Deep Radar

Radar deep learning toolkit.

## Setup

1. Install dependencies:

    ```sh
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```

2. Get data:

    See [red-rover](https://github.com/WiseLabCMU/red-rover/tree/main/processing) for full instructions.

    For each target dataset:
    ```sh
    export SRC=path/to/dataset      # e.g. `radarhd/data` on a network server
    export DST=path/to/destination  # e.g. `data/data` on a local drive

    roverp export -p $SRC -o $DST --metadata
    roverp align -p $DST --mode left
    ```

## Usage

1. Create model in `models`. See existing examples in `models/`.

2. Create config. See examples in `config`.

3. Train:

    ```sh
    python train.py -c your/config/file.yaml -n model/name
    ```

4. Observe with tensorboard:

    ```sh
    tensorboard --logdir=path/to/results --host=0.0.0.0
    ```

## Current Training Command

Base model:
```
nq python train.py -c rxf[small] obj[map] aug[full] data[indoor,outdoor,bike]
```

Fine tune:
```
nq python train.py -c rxf[small] obj[segment] aug[full] data[indoor,outdoor,bike] -b results/small/base -df
```

Resume:
```
nq python train.py -k results/small/base/checkpoints/last.ckpt -n small -v convergence --metric loss/train
```

## Approximate data processing time

- Cartographer Slam: 1:1 (7950X)
- Segmentation: 1:1 (4090)
- FFT/AOA: ???

## Dataset Breakdown

Raw Data: 2.91T
- Camera (`video.avi`): 860G
- Radar (`iq`): 1.5T
- Lidar:
    - Depth (`rng`): 174G
    - Reflectance (`rfl`): 102G
    - Infrared (`nir`): 274G
- IMU:
    - Acceleration (`acc`): 121M
    - Rotation (`rot`): 121M
    - Angular Acceleration (`avel`): 121M

Metadata: 224M
- Timestamps (`ts`): 154M
- Indices (`*_i.npz`): 70M

Processed Data: 4.13G
- Poses (`pose.npz`): 910M
- Segmentation Maps: 2.1G
- CFAR mask: 1.1G
- AOA estimates: 18G
