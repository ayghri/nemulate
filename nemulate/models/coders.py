from typing import List
from .layers.convs import EarthConv2d
from torch import nn
import torch.nn.functional as F
from .utils import get_layer


class PixelBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, act="silu", norm=True, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, k, padding="same")
        self.norm1 = (
            nn.GroupNorm(min(32, c_out), c_out) if norm else nn.Identity()
        )
        self.conv2 = nn.Conv2d(c_out, c_out, k, padding="same")
        self.norm2 = (
            nn.GroupNorm(min(32, c_out), c_out) if norm else nn.Identity()
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = get_layer(act, inplace=True)

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.activation(self.norm2(self.conv2(x)))
        return x


class Block(nn.Module):
    def __init__(self, c_in, c_out, k=3, act="silu", norm=True, dropout=0.0):
        super().__init__()
        self.conv1 = EarthConv2d(c_in, c_out, k)
        self.norm1 = (
            nn.GroupNorm(min(32, c_out), c_out) if norm else nn.Identity()
        )
        self.conv2 = EarthConv2d(c_out, c_out, k)
        self.norm2 = (
            nn.GroupNorm(min(32, c_out), c_out) if norm else nn.Identity()
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = get_layer(act, inplace=True)

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.activation(self.norm2(self.conv2(x)))
        return x


class Down(nn.Module):
    """Stride-2 EarthConv keeps boundary semantics consistent."""

    def __init__(self, c_in, c_out, expansion=2, k=3, act="silu", stride=2):
        super().__init__()
        self.conv1 = EarthConv2d(c_in, c_out, k, stride=stride)
        self.norm1 = nn.GroupNorm(min(32, c_out), c_out)

        self.conv2 = EarthConv2d(c_out, c_out * expansion, k, stride=1)
        self.norm2 = nn.GroupNorm(min(32, c_out * expansion), c_out * expansion)
        self.down = nn.Sequential(
            EarthConv2d(c_in, c_out * expansion, k, stride=stride),
            nn.GroupNorm(min(32, c_out * expansion), c_out * expansion),
        )

        self.act = get_layer(act, inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + self.down(residual))


class Up(nn.Module):
    """Nearest upsample + EarthConv (simple & robust)."""

    def __init__(self, c_in, c_out, k=3, act="silu"):
        super().__init__()
        self.conv = EarthConv2d(c_in, c_out, k)
        self.norm = nn.GroupNorm(min(32, c_out), c_out)
        self.act = get_layer(act, inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.act(self.norm(self.conv(x)))


class EarthEncoder(nn.Module):
    def __init__(self, widths: List[int], activation="SiLU", dropout=0.05):
        super().__init__()


class EarthAE(nn.Module):
    """
    Plain spherical autoencoder (no UNet skips).
    H, W must be divisible by 2**D where D = len(widths)-1.
    """

    expansion = 2

    def __init__(
        self,
        input_shape=(5, 180, 360),
        width_base=32,
        num_layers=4,
        k: int = 3,
        include_land_mask: bool = False,
        bottleneck_dropout=0.1,
    ):
        super().__init__()
        # Encoder

        in_ch = input_shape[0]
        map_size = input_shape[1:]

        c = in_ch
        w = width_base
        self.land_encoder = nn.Sequential()

        if include_land_mask:
            self.land_encoder = PixelBlock(in_ch, in_ch, k=3)

        encoder_layers = []
        for i in range(num_layers - 1):
            encoder_layers.append(Block(c, w, k=k))
            encoder_layers.append(
                Down(w, w, k=k, expansion=self.expansion, stride=2)
            )
            c = w * self.expansion
            w = w * self.expansion

        encoder_layers.append(Block(c, w, k=k, dropout=bottleneck_dropout))
        encoder_layers.append(
            Down(w, w, k=k, expansion=self.expansion, stride=1)
        )
        # encoder_layers.append(Block(c, c, k=k, drop=bottleneck_drop))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        width = width_base * self.expansion ** (num_layers - 1)
        decoder_layers: List[nn.Module] = [
            Block(width * self.expansion, width, k=k)
        ]
        for i in range(num_layers - 1):
            decoder_layers.append(Up(width, width // self.expansion, k=k))
            width = width // self.expansion
            decoder_layers.append(Block(width, width, k=k))
        decoder_layers.append(EarthConv2d(width, in_ch, kernel_size=3))
        decoder_layers.append(nn.AdaptiveAvgPool2d(map_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode_land_mask(self, land_mask):
        return self.land_encoder(land_mask)

    def forward(self, x, land_mask=None):
        if land_mask is not None:
            x = x + self.encode_land_mask(land_mask)
        z = self.encoder(x)
        y = self.decoder(z)
        return z, y
