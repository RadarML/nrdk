"""GRTv2 tokenizer, encoder, and decoder."""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from einops import rearrange, reduce, repeat
from jaxtyping import Bool, Float, Shaped
from torch import Tensor, nn

from nrdk.modules import ConvEncoder, ConvNextLayer
from nrdk.types import SpectrumData

from .positions import InputPositionalEncoder


class Tokenizer(nn.Module):
    """GRTv2 Tokenizer.

    Args:
        d_raw: raw data channel dimension.
        d_tok: tokenizer network dimension; nominally `>= d_in`.
        d_model: output channel dimension.
        stages: depth of each stage; each stage is 2x the width of the
            previous.
        downsample: downsample factor for each stage; can be an int or a tuple
            of `(height, width)` downsample factors.
        mask_token: whether this model should support masking; if `False`,
            no mask token is created, and positions are zeroed out instead.
        layer: convolutional layer to use in the encoder; must use a NCHW data
            format.
    """

    def __init__(
        self, d_raw: int = 24, d_tok: int = 48, d_model: int = 384,
        stages: Sequence[int] = (2, 2, 2),
        downsample: Sequence[int | Sequence[int]] | None = None,
        mask_token: bool = True,
        layer: Callable[[int], nn.Module] = ConvNextLayer
    ) -> None:
        super().__init__()

        self.proj_in = nn.Linear(d_raw, d_tok)
        self.encoder = ConvEncoder(
            stages=stages, downsample=downsample, d_in=d_tok, width=d_tok,
            layer=layer)
        self.proj_out = nn.Linear(d_tok * 2**(len(stages) - 1), d_model)

        if mask_token:
            self.mask_token = nn.Parameter(torch.randn(d_tok) * 0.02)
        else:
            self.mask_token = None

    def forward(
        self, x: Float[Tensor, "n t doppler tx rx rng ch"],
        mask: Bool[Tensor, "n t doppler rng"] | None = None
    ) -> Float[Tensor, "n t doppler_tok rng_tok c"]:
        """Tokenize radar spectrum.

        Args:
            x: raw spectrum with antenna and channel axes.
            mask: boolean mask at input spatial resolution; `True` replaces a
                token with the learned mask token after `proj_in` but before
                the convolutional encoder. Broadcast over the time axis.

        Returns:
            Tokens with time, doppler, and range axes preserved.
        """
        n, t = x.shape[:2]

        # 1. Flatten tx/rx/ch and project to token dimension
        x = rearrange(x, 'n t d tx rx r ch -> n t d r (tx rx ch)')
        x = self.proj_in(x)  # (n, t, doppler, rng, d_tok)

        # 2. Apply masking before the conv encoder to prevent leakage
        if mask is not None:
            mask_exp = rearrange(mask, 'n t d r -> n t d r 1')
            if self.mask_token is not None:
                fill = rearrange(self.mask_token, 'c -> 1 1 1 1 c')
            else:
                fill = torch.zeros_like(x)
            x = torch.where(mask_exp, fill, x)

        # 3. Run ConvEncoder on (doppler, rng), batching over n and t
        x = rearrange(x, 'n t d r c -> (n t) c d r')
        x = self.encoder(x)  # (n*t, d_enc, d', r')
        x = rearrange(x, '(n t) c d r -> n t d r c', n=n, t=t)

        # 4. Project to output dimension
        x = self.proj_out(x)  # (n, t, d', r', d_out)

        return x


class Encoder(nn.Module):
    """GRTv2 Encoder.

    Uses a Video-MAE style encoder, where masked tokens are dropped from the
    sequence before the transformer stem.

    - The mask is provided at spatial (input) resolution and converted to token
        resolution with AND semantics: a token is masked only if every spatial
        position that maps to it is masked.
    - If no mask is provided, the encoder acts like a standard transformer
        encoder.

    Args:
        tokenizer: tokenizer stage; must output a NT(DR)C data format.
        positional_encoder: positional encoding stage.
        stem: transformer stem stage; must use a NTSC data format.
        d_model: model embedding dimension; must match the stem width.
        n_registers: number of learnable register tokens appended to the
            sequence after positional encoding.
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        positional_encoder: InputPositionalEncoder,
        stem: nn.Module,
        d_model: int = 384, n_registers: int = 1,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.positional_encoder = positional_encoder
        self.stem = stem
        self.n_registers = n_registers
        self.registers = nn.Parameter(torch.randn(n_registers, d_model) * 0.02)

    def forward(
        self,
        x: Float[Tensor, "n t doppler x1 x2 rng ch"] | SpectrumData,
        t: Float[Tensor, "n t"] | None = None,
        mask: Bool[Tensor, "n t doppler rng"] | None = None,
        strip_registers: bool = False,
    ) -> Float[Tensor, "n t s_vis c"]:
        """Encode radar spectrum, dropping masked tokens from the sequence.

        !!! note

            If provided, masks are converted to token resolution by AND: a
            token is dropped only if all of its spatial positions are masked.

            Every `(n, t)` slice must also yield equal numbers of masked tokens
            after conversion so the output sequence length is consistent across
            the batch.

        Args:
            x: raw spectrum tensor or `SpectrumData`.
            t: timestamps for each frame in seconds.
            mask: boolean mask at spatial (input) resolution; `True` = masked.
                If `None`, all tokens are kept.
            strip_registers: if `True`, strip register tokens from the output
                sequence before returning.

        Returns:
            Encoded visible tokens and register tokens (unless
                `strip_registers = True`).

        Raises:
            ValueError: if `x` is a plain tensor and `t` is not provided.
        """
        if not isinstance(x, Tensor):
            if t is None:
                t = x.timestamps.float()
            x = x.spectrum
        elif t is None:
            raise ValueError(
                "timestamps must be provided separately when not passing "
                "SpectrumData")

        tokens = self.tokenizer(x, mask=mask)  # (n, T, d', r', c)
        tokens = self.positional_encoder(tokens, t)
        n, T, d_tok, r_tok, _ = tokens.shape
        tokens = rearrange(tokens, "n t d r c -> n t (d r) c")  # (n, T, s, c)

        if mask is not None:
            # Convert spatial mask to token resolution with AND semantics:
            # a token is masked iff every contributing spatial cell is masked.
            mask_tok = reduce(
                mask, 'n t (d sd) (r sr) -> n t d r', 'min', d=d_tok, r=r_tok)
            mask_flat = rearrange(mask_tok, "n t d r -> n t (d r)")  # (n, T, s)
            vis = ~mask_flat
            tokens = rearrange(tokens[vis], '(n t s) c -> n t s c', n=n, t=T)

        regs = repeat(self.registers, 's c -> n (t s) c', n=n, t=T)
        t_regs = repeat(t, 'n t -> n (t s)', s=self.n_registers)
        regs = self.positional_encoder(regs, t_regs)
        regs = rearrange(regs, 'n (t s) c -> n t s c', t=T, s=self.n_registers)
        tokens = torch.cat([tokens, regs], dim=2)

        tokens = self.stem(tokens)
        if strip_registers:
            tokens = tokens[:, :, :-self.n_registers]

        return tokens


class Decoder(nn.Module):
    """GRTv2 Decoder.

    Args:
        decoder: transformer decoder stage; must use a NSC data format.
        positional_encoder: positional encoding stage for output queries,
            which maps a query to `(batch, n, d_model)` position tokens;
            nominally an
            [`OutputPositionalEncoder`][nrdk.models.grtv2.OutputPositionalEncoder].
    """

    def __init__(
        self, decoder: nn.Module,
        positional_encoder: nn.Module,
    ) -> None:
        super().__init__()

        self.decoder = decoder
        self.positional_encoder = positional_encoder

    def forward(
        self, x: Float[Tensor, "n t s c"],
        query: Any
    ) -> Shaped[Tensor, "n ..."] | dict[str, Shaped[Tensor, "..."]]:
        """Decode output queries from encoded tokens.

        Args:
            x: encoded tokens, including any register tokens.
            query: output query positions, in the format expected by
                `positional_encoder` (nominally a
                [`Query`][nrdk.models.grtv2.Query]). Each query is positionally
                encoded and used as an input token to the decoder, which
                attends to the (flattened) encoded tokens.

        Returns:
            Decoder output for each query.
        """
        pos = self.positional_encoder(query)
        nsc = rearrange(x, "n t s c -> n (t s) c")
        return self.decoder(pos, nsc)
