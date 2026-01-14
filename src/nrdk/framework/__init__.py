"""Pytorch lightning-based framework.

We supply and standardize the following components:

- [`TokenizerEncoderDecoder`][.]: wraps a tokenizer, encoder, and a decoder
    into a single [`nn.Module`][torch.].
- [`NRDKLightningModule`][.]: a standardized lightning module which takes a
    [`nn.Module`][torch.] (i.e., [`TokenizerEncoderDecoder`][.]) as well
    as the training objectives, dataloader transforms, etc.
"""

from nrdk._typecheck import typechecker

with typechecker("nrdk.framework"):
    from .architectures import Output, TokenizerEncoderDecoder
    from .callbacks import GradientStats, PerformanceMonitor
    from .lightningmodule import NRDKLightningModule
    from .load import load_model
    from .logging import LoggerWithImages, MLFlowLogger, TensorBoardLogger
    from .result import Result

__all__ = [
    "load_model",
    "Output", "TokenizerEncoderDecoder",
    "GradientStats", "PerformanceMonitor",
    "LoggerWithImages", "MLFlowLogger", "TensorBoardLogger",
    "NRDKLightningModule", "Result"]
