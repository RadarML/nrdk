"""GRTv2 positional encoders."""

from collections.abc import Sequence
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn

from nrdk.modules import FourierFeatures

from .query import Query


class InputPositionalEncoder(nn.Module):
    """GRTv2 Input Positional Encoder.

    Args:
        d_model: model embedding dimension; must match the encoder output width.
        d_features: feature dimension of the input tokens; must match the
            output width of the tokenizer.
        coef: base frequency coefficient for the positional encoding.
        scale_t: scale factor for the time positional encoding.
    """

    def __init__(
        self, d_model: int = 384, d_features: int = 32, coef: float = 10000.0,
        scale_t: float = 20.0
    ) -> None:
        super().__init__()
        self.proj_t = nn.Linear(d_features, d_model)
        self.proj_r = nn.Linear(d_features, d_model)
        self.proj_d = nn.Linear(d_features, d_model)
        self.scale_t = scale_t

        self.pos = FourierFeatures(d_features, coef=coef)

    def forward(
        self, x: Float[Tensor, "n t d r c"] | Float[Tensor, "n t c"],
        t: Float[Tensor, "n t"]
    ) -> Float[Tensor, "n t d r c"] | Float[Tensor, "n t c"]:
        """Add positional encodings to tokenized radar spectrum.

        Args:
            x: token embeddings, either full `(doppler, range)` tokens or
                per-frame metadata tokens (e.g. register tokens) with no
                spatial structure.
            t: relative timestamps for each frame, in seconds.

        Returns:
            Token embeddings with positional biases added in-place. Frame
                metadata tokens only receive the temporal bias; full spectrum
                tokens also receive Doppler and range biases.
        """
        # Time: t has shape (n, T); scale and encode -> (n, T, d_model)
        t_enc = self.proj_t(self.pos(t * self.scale_t))

        if x.ndim == 3:
            return x + t_enc

        _, _, d, r, _ = x.shape

        # Doppler: centered around 0 -> (d, d_model)
        d_idx = torch.arange(d, device=x.device, dtype=x.dtype) - d // 2
        d_enc = self.proj_d(self.pos(d_idx))

        # Range: linear indices -> (r, d_model)
        r_idx = torch.arange(r, device=x.device, dtype=x.dtype)
        r_enc = self.proj_r(self.pos(r_idx))

        x = x + t_enc[:, :, None, None, :]
        x = x + d_enc[None, None, :, None, :]
        x = x + r_enc[None, None, None, :, :]

        return x


class OutputPositionalEncoder(nn.Module):
    """GRTv2 Output Query Positional Encoder.

    Args:
        d_model: model embedding dimension; must match the decoder input width.
        d_features: feature dimension of the output tokens; must match the
            input width of the detokenizer.
        coef: base frequency coefficient for the positional encoding.
        scale_t: scale factor for the time positional encoding.
        scale_angle: scale factor for the elevation and azimuth positional
            encoding.
        axes: which axes are encoded by a learnable projection. A query may
            only provide these axes; any other axis must be left unset.
        wildcards: which axes should have learnable wildcard tokens, used in
            place of the positional encoding when a query leaves that axis
            unset.
    """

    def __init__(
        self, d_model: int = 384, d_features: int = 32, coef: float = 10000.0,
        scale_t: float = 20.0, scale_angle: float = 200.0,
        axes: Sequence[
            Literal["t", "rng", "az", "el"]] = ("t", "rng", "az", "el"),
        wildcards: Sequence[Literal["t", "rng", "az", "el"]] = (),
    ) -> None:
        super().__init__()

        for k in (*axes, *wildcards):
            if k not in ("t", "rng", "az", "el"):
                raise ValueError(
                    f"Invalid axis key: {k}. Supported axes are 't', "
                    "'rng', 'az', and 'el'.")

        self.project = nn.ModuleDict({
            k: nn.Linear(d_features, d_model) for k in axes})
        self.wildcards = nn.ParameterDict({
            k: nn.Parameter(torch.randn(d_model) * 0.02)
            for k in wildcards})

        self.scale_t = scale_t
        self.scale_angle = scale_angle

        self.pos = FourierFeatures(d_features, coef=coef)

        # Returned when no axis contributes; broadcast against the query, in
        # the same way as a wildcard token.
        self.register_buffer(
            "empty", torch.zeros(1, 1, d_model), persistent=False)

    def _encode(
        self, x: Float[Tensor, "batch n"] | None,
        name: str, scale: float = 1.0,
    ) -> Float[Tensor, "batch n d_model"] | None:
        if x is None:
            if name not in self.wildcards:
                return None
            return self.wildcards[name][None, None, :]

        if name not in self.project:
            raise ValueError(
                f"Query provides the '{name}' axis, but this positional "
                "encoder was not configured to encode it; `axes` is "
                f"{tuple(self.project.keys())}.")
        return self.project[name](self.pos(x * scale))

    def forward(
        self, query: Query[Tensor]
    ) -> Float[Tensor, "batch n d_model"]:
        """Calculate positional encodings for output queries.

        Args:
            query: output query specifying positions along each dimension.
                `None` fields are replaced with a learnable wildcard token if
                one is configured for that axis, and are otherwise omitted.

        Returns:
            Position tokens for each output query, summed over every axis
                which contributes. Zero if all axes are `None` and no wildcards
                are configured.

        Raises:
            ValueError: the query provides an axis which is not in `axes`.
        """
        encoded = [
            self._encode(query.t, "t", self.scale_t),
            self._encode(query.rng, "rng"),
            self._encode(query.az, "az", self.scale_angle),
            self._encode(query.el, "el", self.scale_angle)]
        present = [x for x in encoded if x is not None]

        if len(present) == 0:
            return self.empty  # type: ignore
        return sum(present[1:], start=present[0])
