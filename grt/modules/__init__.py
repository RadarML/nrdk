"""Reusable modules with radar-specific modifications.

.. [M1] RoFormer: Enhanced Transformer with Rotary Position Embedding
    https://arxiv.org/abs/2104.09864
.. [M2] ConvNeXt: A ConvNet for the 2020s
    https://arxiv.org/abs/2201.03545
.. [M3] On Layer Normalization in the Transformer Architecture
    https://arxiv.org/pdf/2002.04745.pdf
.. [M4] Issue @ pytorch relating to post-norm:
    https://github.com/pytorch/pytorch/issues/55270
.. [M5] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    https://arxiv.org/abs/2103.14030
.. [M6] Swin Transformer - Implementation (see `models/swin_transformer_v2.py`)
    https://github.com/microsoft/Swin-Transformer
.. [M7] Vision Transformers for Dense Prediction
    https://arxiv.org/abs/2103.13413
.. [M8] Axial Attention in Multidimensional Transformers
    https://arxiv.org/abs/1912.12180
"""

from .conv import ConvDownsample, ConvNeXTBlock, ConvResidual, ConvSeparable
from .dpt import Fusion2D, FusionDecoder
from .patch import FFTLinear, Patch2D, Patch4D, PatchMerge, Unpatch
from .position import Learnable1D, LearnableND, Readout, Rotary2D, Sinusoid
from .swin import AxialTransformerLayer, SwinTransformerLayer, WindowAttention
from .transformer import BasisChange, TransformerDecoder, TransformerLayer
from .window import RelativePositionBias, WindowPartition

__all__ = [
    "ConvDownsample", "ConvNeXTBlock", "ConvResidual", "ConvSeparable",
    "Fusion2D", "FusionDecoder",
    "FFTLinear", "Patch2D", "Patch4D", "Unpatch", "PatchMerge",
    "Learnable1D", "LearnableND", "Readout", "Rotary2D", "Sinusoid",
    "AxialTransformerLayer", "SwinTransformerLayer", "WindowAttention",
    "BasisChange", "TransformerDecoder", "TransformerLayer",
    "RelativePositionBias", "WindowPartition"
]
