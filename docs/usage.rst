Usage
=====

1. Install dependencies:

   .. code-block:: sh

      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      pip install -r requirements.txt

2. Get data:

   See `red-rover <https://github.com/WiseLabCMU/red-rover/tree/main/processing>`_ for full instructions. For each target dataset:

   .. code-block:: sh

      export SRC=path/to/dataset      # e.g. `radarhd/data` on a network server
      export DST=path/to/destination  # e.g. `data/data` on a local drive

      roverp export -p $SRC -o $DST --metadata
      roverp align -p $DST --mode left
      roverp decompress -p $SRC -o $DST
      cp $SRC/radar/iq $DST/radar/iq

3. Train:
   
   .. code-block:: sh

      python train.py -c your/config/file.yaml -n model_name -v version

   - `model_name` and `version` should be human-readable names; they must be
     valid unix file names (e.g. no `/`)

4. Observe with tensorboard:

   .. code-block:: sh
      
      tensorboard --logdir=path/to/results --host=0.0.0.0 --samples_per_plugin=images=100



Configuration
-------------

Model configurations are specified using a yaml file with the following format:

.. code-block:: yaml

   objective: ObjectiveName
   model: ModelName
   model_args: {...}
   optimizer: OptimizerName
   optimier_args: {...}
   dataset: {...}
   ... other objective args

These values only include model/method specifications, and are recorded as `hparams.yaml` when the model is trained. Note that environment-specific parameters such as stopping conditions, logging intervals, file paths, etc. are not provided via `config.yaml`, and not recorded in `hparams.yaml`.

- `objective`: name of the training objective to use; see :doc:`objectives`. All other entries in `config.yaml` are passed to the selected `Objective`.
- `model`, `model_args`: name of the model class to use, and constructor arguments for that class; see :doc:`models`.
- `optimizer`, `optimizer_args`: optimizer to use, and constructor arguemnts; the `optimizer` should be a member of `torch.optim <https://pytorch.org/docs/stable/optim.html>`_.
- `dataset`: dataset configuration options (passthrough to :class:`.RoverDataModule`); see :doc:`dataloader` for details.


Training Script
---------------

.. argparse::
   :func: _parse
   :filename: ../train.py
   :prog: train.py
