r"""Deepradar: Deep Learning Toolkit for Radar

      |                               |
   _` |  _ \  _ \ __ \   __| _` |  _` |  _` |  __|
  (   |  __/  __/ |   | |   (   | (   | (   | |
 \__,_|\___|\___| .__/ _|  \__,_|\__,_|\__,_|_|
                 _|

"""  # noqa: D208

from jaxtyping import install_import_hook

with install_import_hook("deepradar", "beartype.beartype"):
    from . import (
        augmentations,
        config,
        dataloader,
        modules,
        objectives,
        transforms,
    )
    from .deepradar import DeepRadar


__all__ = [
    "dataloader", "transforms", "augmentations", "modules", "objectives",
    "config", "DeepRadar"]
