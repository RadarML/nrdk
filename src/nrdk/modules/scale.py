"""Range-Doppler magnitude scaling."""

import abc

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class Scale(nn.Module, abc.ABC):
    """Range-Doppler magnitude scaling.

    The magnitude of the spectrum vector associated with each range-Doppler bin
    is modified via exponentiation, keeping the direction unchanged.
    """

    @abc.abstractmethod
    def forward(
        self, spec: Float[Tensor, "batch t doppler tx rx rng ch"],
        reverse: bool = False
    ) -> Float[Tensor, "batch t doppler tx rx rng ch"]:
        """Apply normalization.

        Args:
            spec: input spectrum data.
            reverse: if `True`, apply inverse scaling to recover original
                magnitudes.

        Returns:
            Scaled spectrum data.
        """
        ...


class Power(Scale):
    """Range-Doppler magnitude scaling by raising to a power.

    For power `p`, each range doppler bin is normalized as:
    ```
    x' = (x / ||x||_2) * ||x||_2^p = x * ||x||_2^(p - 1)
    ```
    Or, in reverse:
    ```
    x = (x' / ||x'||_2) * ||x'||_2^(1/p) = x' * ||x'||_2^(1/p - 1)
    ```

    Args:
        power: exponent for the magnitude.
    """

    def __init__(self, power: float = 1.0) -> None:
        super().__init__()
        self.power = power

    def forward(
        self, spec: Float[Tensor, "batch t doppler tx rx rng ch"],
        reverse: bool = False
    ) -> Float[Tensor, "batch t doppler tx rx rng ch"]:
        rd = rearrange(spec, "b t d tx rx rng ch -> b t d rng (tx rx ch)")
        norm = torch.linalg.norm(rd, dim=-1)

        if reverse:
            exp = 1 / self.power - 1
        else:
            exp = self.power - 1
        power = norm ** exp
        power = torch.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
        spec = spec * power[:, :, :, None, None, :, None]

        return spec


class Rational(Scale):
    """Rational magnitude scaling.

    For coefficient `c`, the magnitude is scaled as
    ```
    x' = x / (||x||_2 + c)
    ```
    Or, in reverse:
    ```
    x = x' * c / (1 - ||x'||_2)
    ```

    Args:
        coef: rational denominator coefficient `c`.
    """

    def __init__(self, coef: float = 0.05) -> None:
        super().__init__()
        self.coef = coef

    def forward(
        self, spec: Float[Tensor, "batch t doppler tx rx rng ch"],
        reverse: bool = False
    ) -> Float[Tensor, "batch t doppler tx rx rng ch"]:
        rd = rearrange(spec, "b t d tx rx rng ch -> b t d rng (tx rx ch)")
        norm = torch.linalg.norm(rd, dim=-1)

        if reverse:
            scale = self.coef / (1 - norm)
        else:
            scale = 1 / (norm + self.coef)

        scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
        spec = spec * scale[:, :, :, None, None, :, None]

        return spec


class ASinh(Scale):
    """Hyperbolic sine magnitude scaling.

    For scale parameter `s`, the magnitude is scaled as:
    ```
    x' = x * asinh(||x||_2 * s) / (||x||_2 * s)
    ```
    Or, in reverse:
    ```
    x = x' * sinh(||x'||_2 * s) / (||x'||_2 * s)
    ```

    This provides log-like compression for large values while remaining linear
    near zero. The transformation is perfectly invertible with no singularities.

    Args:
        scale: scaling parameter `s`. Larger values compress more aggressively.
    """

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(
        self, spec: Float[Tensor, "batch t doppler tx rx rng ch"],
        reverse: bool = False
    ) -> Float[Tensor, "batch t doppler tx rx rng ch"]:
        rd = rearrange(spec, "b t d tx rx rng ch -> b t d rng (tx rx ch)")
        norm = torch.linalg.norm(rd, dim=-1)

        # Small epsilon to prevent division by zero
        eps = 1e-8
        norm_scaled = norm * self.scale

        if reverse:
            # Recover original magnitude: sinh(asinh(x)) = x
            factor = torch.sinh(norm_scaled) / (norm_scaled + eps)
        else:
            # Compress magnitude with asinh
            factor = torch.asinh(norm_scaled) / (norm_scaled + eps)

        factor = torch.nan_to_num(factor, nan=1.0, posinf=1.0, neginf=1.0)
        spec = spec * factor[:, :, :, None, None, :, None]

        return spec


class Mixed(Scale):
    """Combine a different forward and reverse scale.

    Args:
        forward: forward scaling.
        reverse: reverse scaling.
    """

    def __init__(self, forward: Scale | None, reverse: Scale | None) -> None:
        super().__init__()
        self.forward_scale = forward
        self.reverse_scale = reverse

    def forward(
        self, spec: Float[Tensor, "batch t doppler tx rx rng ch"],
        reverse: bool = False
    ) -> Float[Tensor, "batch t doppler tx rx rng ch"]:
        if reverse:
            if self.reverse_scale is None:
                return spec
            return self.reverse_scale(spec)
        else:
            if self.forward_scale is None:
                return spec
            return self.forward_scale(spec)
