import torch
from einops import rearrange
from radar import modules
from torch import nn


class RadarTransformer(nn.Module):

    def __init__(
        self, dim: int = 768, ff: int = 3072,
        heads: int = 12, dropout: float = 0.1
    ) -> None:
        super().__init__()

        # self.patch = modules.Patch4D(
        #    channels=2, features=dim, size=(16, 16))
        # self.patch = modules.Patch2D(
        #     channels=2 * 64, features=512, size=(1, 8))
        self.patch = modules.Patch2D(
            channels=2 * 2 * 64, features=dim, size=(1, 2))

        self.pos = modules.Sinusoid()
        # self.pos = modules.Learnable1D(d_model=512, size=1024)
        # self.pos = modules.LearnableND(d_model=512, shape=(4, 16, 8, 2))
        # self.pos = modules.LearnableND(d_model=dim, shape=(8, 256 // 2))

        self.encode = nn.Sequential(*[
            modules.TransformerLayer(
                d_model=dim, n_head=heads, d_feedforward=ff, dropout=dropout,
                activation=torch.nn.GELU())
            for _ in range(2)])
    
        # self.query = modules.BasisChange(
        #     d_model=dim, n_head=heads, shape=(1024 // 16, 256 // 16))

        # self.decode = nn.Sequential(*[
        #     modules.TransformerLayer(
        #         d_model=dim, n_head=heads, d_feedforward=ff, dropout=dropout,
        #         activation=torch.nn.GELU())
        #     for _ in range(4)])

        self.unpatch = modules.Unpatch2D(
            output_size=(1024, 256, 1), features=dim, size=(16, 16))

        self.activation = nn.Sigmoid()

    def forward(self, x):

        # patch = self.patch(rearrange(x, "n d a e r c -> n c d r a e"))
        patch = self.patch(rearrange(x, "n d a e r c -> n (d c e) a r"))
        # patch = self.patch(rearrange(x, "n d a r c -> n (d c) a r"))

        embedded = self.pos(patch)
        enc = self.encode(
            rearrange(embedded, "n a r c -> n (a r) c"))
        # enc = self.encode(
        #     rearrange(embedded, "n d r a e c -> n (d r a e) c"))
        # tf = self.query(enc)
        tf = enc
        dec = tf
        # dec = self.decode(tf)
        # dec = self.encode(
        #     rearrange(embedded, "n a r c -> n (a r) c"))
        # dec = self.encode(rearrange(patch, "n a r c -> n (a r) c"))

        unpatch = self.unpatch(dec)[:, 0]
        return self.activation(unpatch)
