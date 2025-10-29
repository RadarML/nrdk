"""Pytorch lightning-based framework.

We supply and standardize the following components:

- [`TokenizerEncoderDecoder`][.]: wraps a tokenizer, encoder, and a decoder
    into a single [`nn.Module`][torch.].
- [`NRDKLightningModule`][.]: a standardized lightning module which takes a
    [`nn.Module`][torch.] (i.e., [`TokenizerEncoderDecoder`][.]) as well
    as the training objectives, dataloader transforms, etc.
"""

from jaxtyping import install_import_hook

with install_import_hook("nrdk.framework", "beartype.beartype"):
    from .architectures import Regularized, TokenizerEncoderDecoder
    from .callbacks import PerformanceMonitor
    from .lightningmodule import NRDKLightningModule
    from .logging import LoggerWithImages, MLFlowLogger, TensorBoardLogger
    from .result import Result

__all__ = [
    "Regularized", "TokenizerEncoderDecoder",
    "PerformanceMonitor",
    "LoggerWithImages", "MLFlowLogger", "TensorBoardLogger",
    "NRDKLightningModule", "Result"]
