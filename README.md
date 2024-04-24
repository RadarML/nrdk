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
    roverp decompress -p $SRC -o $DST
    cp $SRC/radar/iq $DST/radar/iq
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
