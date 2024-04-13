"""RadarML modules."""

import torch
from torch import nn, Tensor, fft
from jaxtyping import Float32, Complex64


class FFT2Pad(nn.Module):
    """Padded range-antenna FFT.
    
    Parameters
    ----------
    pad: padding to apply.
    """

    def __init__(self, pad: int = 8) -> None:
        super().__init__()
        self.pad = pad

    def forward(
        self, x: Complex64[Tensor, "batch doppler tx rx range"]
    ) -> Float32[Tensor, "batch doppler_iq azimuth range"]:
        iq_dar = x.reshape(x.shape[0], x.shape[1], -1, x.shape[4])
        zeros = torch.zeros(
            x.shape[0], x.shape[1], self.pad, x.shape[4], device=x.device)

        iq_pad = torch.concatenate([iq_dar, zeros], dim=2)
        dar = fft.fftn(iq_pad, dim=(2, 3))
        dar_shf = fft.fftshift(dar, dim=2)

        dra = torch.swapaxes(dar_shf, 2, 3)
        return torch.concatenate(
            [torch.abs(dra) / 1e6, torch.angle(dra)], dim=1)


class Rotary2D(nn.Module):
    """2D rotary encoding (RoPE).
    
    https://arxiv.org/abs/2104.09864

    Parameters
    ----------
    features: number of features in this embedding. 
    """

    def __init__(self, features: int = 512) -> None:
        super().__init__()
        self.d = features // 2

    def forward(
        self, x: Float32[Tensor, "b x1 x2 f"]
    ) -> Float32[Tensor, "b x1 x2 f"]:

        # Eq. 15. Extra zero since our dims are relatively small.
        i = torch.arange(x.shape[3] // 4, device=x.device)
        theta = 100000 ** (-2 * (i - 1) / x.shape[1] // 2)
        m1 = torch.arange(x.shape[1], device=x.device)
        m2 = torch.arange(x.shape[2], device=x.device)

        def _get_R(m) -> Float32[torch.Tensor, "x d 2 2"]:
            mt = m[:, None] * theta[None, :]
            cos = torch.cos(mt)
            sin = torch.sin(mt)
            R_stack = torch.stack(
                [torch.stack([cos, -sin]), torch.stack([sin, cos])])
            return torch.moveaxis(R_stack, (0, 1, 2, 3), (2, 3, 0, 1))

        x1: Float32[torch.Tensor, "b x1 x2 d 2 1"] = (
            x[..., :self.d].reshape(*x.shape[:-1], -1, 2)[..., None])
        R1: Float32[torch.Tensor, "1 x1  1 d 2 2"] = (
            _get_R(m1)[None, :, None, :, :, :])
        x1_emb = torch.matmul(R1, x1).reshape(*x.shape[:-1], -1)

        x2: Float32[torch.Tensor, "b x1 x2 d 2 1"] = (
            x[..., self.d:].reshape(*x.shape[:-1], -1, 2)[..., None])
        R2: Float32[torch.Tensor, "1 1  x2 d 2 2"] = (
            _get_R(m2)[None, None, :, :, :, :])
        x2_emb = torch.matmul(R2, x2).reshape(*x.shape[:-1], -1)

        return torch.concatenate([x1_emb, x2_emb], dim=3)


class Patch(nn.Module):
    """Split data into patches, and apply a 2D positional embedding.
    
    Patches are arranged in doppler-range order.

    Parameters
    ----------
    in_channels: number of input channels (tx * rx * 2 for complex data).
    features: number of output features; should be `>= in_channels * size**2`.
    size: patch size.
    """

    def __init__(
        self, channels: int = 1, features: int = 512,
        size: tuple[int, int] = (4, 4)
    ) -> None:
        super().__init__()
        self.size = size
        self.patch_features = channels * size[0] * size[1]
        self.features = features

        self.patch = nn.Unfold(kernel_size=size, stride=size)
        self.linear = nn.Linear(self.patch_features, features)
        self.rotary = Rotary2D(features)
        
    def forward(
        self, x: Float32[Tensor, "b f1 x1 x2"]
    ) -> Float32[Tensor, "b p f2"]:

        patches: Float32[Tensor, "b f1p p"] = self.patch(x)

        patches_batch: Float32[Tensor, "bb f1p"] = (
            torch.moveaxis(patches, 1, 2).reshape(-1, patches.shape[1]))
        embedding_batch: Float32[Tensor, "bb f2"] = self.linear(patches_batch)
        embedding: Float32[Tensor, "b x1p x2p f2"] = embedding_batch.reshape(
            x.shape[0], x.shape[2] // self.size[0],
            x.shape[3] // self.size[1], self.features)

        embedding_rot: Float32[Tensor, "b x1p x2p f2"] = self.rotary(embedding)
        return embedding_rot.reshape(x.shape[0], -1, self.features)


class Unpatch(nn.Module):
    """Unpatch data.
    
    Patches are arranged in azimuth-range order.

    Parameters
    ----------
    output_size: (azimuth, range) bins.
    features: number of input features; should be `>= size * size`.
    size: patch size.
    """

    def __init__(
        self, output_size: tuple[int, int] = (1024, 256), features: int = 512,
        size: tuple[int, int] = (16, 16)
    ) -> None:
        super().__init__()
        self.features = features
        self.size = size
        self.output_size = output_size

        self.linear = nn.Linear(features, size[0] * size[1])
        self.unpatch = nn.Fold(output_size, kernel_size=size, stride=size)

    def forward(
        self, x: Float32[Tensor, "b p f"]
    ) -> Float32[Tensor, "b x1 x2"]:
        patch_bps: Float32[Tensor, "b p s"] = self.linear(x).reshape(
            x.shape[0], -1, self.size[0] * self.size[1])
        patch_bsp: Float32[Tensor, "b s p"] = torch.swapaxes(patch_bps, 1, 2)
        unpatch: Float32[Tensor, "b 1 x1 x2"] = self.unpatch(patch_bsp)
        return unpatch.reshape(x.shape[0], *self.output_size)


class BasisTransform(nn.Module):
    """Attention-based change of basis."""

    def __init__(
        self, output_size: tuple[int, int] = (64, 16), features: int = 512
    ) -> None:
        self.output_size = output_size
        self.features = features
