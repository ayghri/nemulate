import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from nemulate.models.layers.convs import EarthConv2d


class InvertedEarthResidual(nn.Module):
    """
    Merged Block: Handles Stride + Expansion (c -> 4c -> out_c) + Residual.
    1. Expand (1x1)
    2. Spatial Conv (3x3) [Handles Stride]
    3. Project (1x1)
    """

    def __init__(self, in_c, out_c, stride=1, padding=1, expand_ratio=4):
        super().__init__()
        self.stride = stride
        hidden_dim = in_c * expand_ratio

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, hidden_dim, 1, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.SiLU(inplace=True),
            EarthConv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride=stride,
                padding=padding,
                groups=hidden_dim,
                bias=False,
            ),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
        )

        self.shortcut = nn.Identity()
        if isinstance(padding, int):
            padding = (padding, padding)
        padding = tuple(p - 1 for p in padding)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                EarthConv2d(
                    in_c, out_c, 1, stride=stride, padding=padding, bias=False
                ),
                nn.InstanceNorm2d(out_c, affine=True),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class InvertedResidual(nn.Module):
    """
    Merged Block: Handles Stride + Expansion (c -> 4c -> out_c) + Residual.
    1. Expand (1x1)
    2. Spatial Conv (3x3) [Handles Stride]
    3. Project (1x1)
    """

    def __init__(self, in_c, out_c, stride=1, expand_ratio=4, padding=0):
        super().__init__()
        self.stride = stride
        hidden_dim = in_c * expand_ratio

        # 1. Main Path (Wide)
        self.conv = nn.Sequential(
            # Expansion (1x1)
            nn.Conv2d(in_c, hidden_dim, 1, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.SiLU(inplace=True),
            # Spatial Processing (3x3) - Handles Stride here
            # Groups=hidden_dim makes this a "Depthwise" conv (efficient for large expansion)
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                stride=stride,
                padding=padding,
                groups=hidden_dim,
                bias=False,
            ),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.SiLU(inplace=True),
            # Projection (1x1) - Back to output dim
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            # Note: No activation after projection in inverted residuals!
        )

        # 2. Shortcut Path (Skip Connection)
        # If dimensions change or we stride, we need to adapt the residual x
        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c or padding == 0:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_c, out_c, 3, stride=stride, padding=padding, bias=False
                ),
                nn.InstanceNorm2d(out_c, affine=True),
            )

    def forward(self, x):
        # x + self.conv(x)
        # if self.stride == 1 and x.shape[1] == self.conv[-2].out_channels
        # else self.shortcut(x) + self.conv(x)
        return self.conv(x) + self.shortcut(x)


class UpSampler(nn.Module):
    def __init__(self, c, factor=2):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c * factor**2, kernel_size=1),
            nn.PixelShuffle(upscale_factor=factor),
        )

    def forward(self, x):
        return self.up(x)


class InvertedEarthAE(nn.Module):
    def __init__(
        self,
        input_channels=5,
        include_land_mask: bool = False,
        land_mask_channels: int = 1,
    ):
        super().__init__()
        # self.padder = SafePadWrapper(factor=8)
        # self.padder = nn.Identity()

        # --- ENCODER ---
        # Stem: 5 -> 64 (Full Res)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, 1, padding=0),
            nn.InstanceNorm2d(64, affine=True),
            nn.SiLU(inplace=True),
        )

        self.land_encoder = nn.Sequential()

        if include_land_mask:
            self.land_encoder = nn.Conv2d(land_mask_channels, 64, 1, padding=0)

        # Stage 1: 64 -> 128 (Downsample)
        # self.enc1 = InvertedResidual(64, 128, stride=2, expand_ratio=4)
        self.enc1 = InvertedEarthResidual(64, 128, stride=2, padding=(13, 19))

        # Stage 2: 128 -> 256 (Downsample)
        self.enc2 = InvertedResidual(128, 256, stride=2, expand_ratio=4)

        # Stage 3: 256 -> 256 (Downsample to Bottleneck)
        # Stride 2 here takes [48, 96] -> [24, 48]
        # We output to the bottleneck. NO Activation/Norm on the final latent logic.
        self.enc3 = InvertedResidual(256, 256, stride=2, expand_ratio=4)

        # The pure latent projection (removes the Norm/Residual bias from the last block)
        self.to_latent = nn.Conv2d(256, 256, 1)

        # --- DECODER ---

        # Stage 3 Up: [24, 48] -> [48, 96]
        self.up3 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            UpSampler(256, factor=2),
            InvertedResidual(256, 256, stride=1, padding=0),
        )

        # Stage 2 Up: [48, 96] -> [96, 192]
        self.up2 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            UpSampler(256, factor=2),
            InvertedResidual(256, 128, stride=1, padding=1),
        )

        # Stage 1 Up: [96, 192] -> [192, 384]
        self.up1 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            UpSampler(128, factor=2),
            InvertedResidual(128, 64, stride=1, padding=0),
        )

        # Final Head
        self.head = nn.Conv2d(64, 5, 3)

    def encode_land_mask(self, land_mask):
        return self.land_encoder(land_mask)

    def forward(self, x, land_mask=None):
        # Pad
        # x_pad, original_size = self.padder(x)
        original_size = x.shape[2], x.shape[3]

        # Encode
        e0 = self.stem(x)  # [64, 192, 384]

        if land_mask is not None:
            e0 = e0 + self.encode_land_mask(land_mask)

        e1 = self.enc1(e0)  # [128, 96, 192]
        e2 = self.enc2(e1)  # [256, 48, 96]
        e3 = self.enc3(e2)  # [256, 24, 48]

        # Bottleneck
        z = self.to_latent(e3)

        # Decode
        d3 = self.up3(z)  # [256, 48, 96]
        d2 = self.up2(d3)  # [128, 96, 192]
        d1 = self.up1(d2)  # [64, 192, 384]

        # Output
        out = self.head(d1)

        # Crop
        diff_h = out.shape[2] - original_size[0]
        diff_w = out.shape[3] - original_size[1]
        out = out[
            :,
            :,
            diff_h // 2 : out.shape[2] - (diff_h - diff_h // 2),
            diff_w // 2 : out.shape[3] - (diff_w - diff_w // 2),
        ]
        return out, z


# Verification
# model = EarthAE_Wide().cuda()
# x = torch.randn(2, 5, 180, 360).cuda()

# summary(model, input_data=x)
# Check bottleneck shape
# z = model.to_latent(
#     model.enc3(model.enc2(model.enc1(model.stem(model.padder(x)[0]))))
# )
# print(f"Input: {x.shape}")
# print(f"Bottleneck: {z.shape}")  # Expect [2, 256, 24, 48]
# print(f"Output: {model(x).shape}")
