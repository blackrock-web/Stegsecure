"""Official Paper 1 architecture extracted from:
https://github.com/Ayshcoder987/Joint-Encryption-Steganography-CNN

Source file: joint_encryption_steganography_cnn.py (authors' official implementation).
NO pretrained weights are shipped in that repository (Releases: none).
This module provides architecture only for optional user-supplied checkpoints.
"""
from __future__ import annotations

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass

if HAS_TORCH:
    class ResidualBlock(nn.Module):
        """Improved residual block with better gradient flow"""
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                  stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            # Shortcut connection
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out += self.shortcut(residual)
            out = self.relu(out)

            return out

    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DownBlock, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.conv(x)
            pooled = self.pool(x)
            return pooled

    class UpBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(UpBlock, self).__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),  # *2 for skip connection
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x, skip):
            x = self.up(x)

            # Handle dimension mismatches with interpolation
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            x = self.conv(x)
            return x

    class EnhancedEncoder(nn.Module):
        """Enhanced encoder with proper skip connection handling"""
        def __init__(self, in_channels=6, base_channels=64):
            super(EnhancedEncoder, self).__init__()

            # Initial convolution
            self.initial_conv = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )

            # Downsampling blocks
            self.down1 = DownBlock(base_channels, base_channels * 2)
            self.down2 = DownBlock(base_channels * 2, base_channels * 4)
            self.down3 = DownBlock(base_channels * 4, base_channels * 8)

            # Upsampling blocks
            self.up1 = UpBlock(base_channels * 8, base_channels * 4)
            self.up2 = UpBlock(base_channels * 4, base_channels * 2)
            self.up3 = UpBlock(base_channels * 2, base_channels)

            # Final convolution
            self.final_conv = nn.Sequential(
                nn.Conv2d(base_channels, 3, 3, padding=1),
                nn.Tanh()
            )

        def forward(self, x):
            # Initial convolution
            x1 = self.initial_conv(x)

            # Downsampling
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)

            # Upsampling with skip connections
            x = self.up1(x4, x3)  # Pass skip connection
            x = self.up2(x, x2)   # Pass skip connection
            x = self.up3(x, x1)   # Pass skip connection

            # Final convolution
            x = self.final_conv(x)
            return x

    class EnhancedDecoder(nn.Module):
        """Enhanced decoder with proper skip connection handling"""
        def __init__(self, in_channels=3, base_channels=64):
            super(EnhancedDecoder, self).__init__()

            # Initial convolution
            self.initial_conv = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )

            # Downsampling blocks
            self.down1 = DownBlock(base_channels, base_channels * 2)
            self.down2 = DownBlock(base_channels * 2, base_channels * 4)
            self.down3 = DownBlock(base_channels * 4, base_channels * 8)

            # Upsampling blocks
            self.up1 = UpBlock(base_channels * 8, base_channels * 4)
            self.up2 = UpBlock(base_channels * 4, base_channels * 2)
            self.up3 = UpBlock(base_channels * 2, base_channels)

            # Final convolution
            self.final_conv = nn.Sequential(
                nn.Conv2d(base_channels, 3, 3, padding=1),
                nn.Tanh()
            )

        def forward(self, x):
            # Initial convolution
            x1 = self.initial_conv(x)

            # Downsampling
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)

            # Upsampling with skip connections
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)

            # Final convolution
            x = self.final_conv(x)
            return x

    class EnhancedKeyMixer(nn.Module):
        """Improved KeyMixer with adaptive transformation"""
        def __init__(self, channels=3):
            super(EnhancedKeyMixer, self).__init__()

            # Adaptive transformation network
            self.transformation = nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, channels, kernel_size=3, padding=1),
                nn.Tanh()  # Better than Sigmoid for transformation
            )

            # Learnable parameters
            self.alpha = nn.Parameter(torch.tensor(0.1))
            self.beta = nn.Parameter(torch.tensor(0.1))

        def forward(self, cover, secret):
            # Transform the secret
            transformed_secret = self.transformation(secret)

            # Adaptive mixing
            mixed_secret = self.alpha * transformed_secret + self.beta * secret

            # Concatenate with cover
            combined = torch.cat([cover, mixed_secret], dim=1)
            return combined


    class OfficialPaper1Model(nn.Module):
        """Wrapper: KeyMixer + Encoder + Decoder matching official script."""

        def __init__(self):
            super().__init__()
            self.mixer = EnhancedKeyMixer()
            self.encoder = EnhancedEncoder()
            self.decoder = EnhancedDecoder()

        def embed(self, cover, secret):
            # cover, secret in [-1,1] or [0,1]; official uses Tanh outputs
            combined = self.mixer(cover, secret)  # cat cover + mixed secret
            container = self.encoder(combined)
            return container

        def extract(self, container):
            return self.decoder(container)

else:
    OfficialPaper1Model = None  # type: ignore
    EnhancedEncoder = EnhancedDecoder = EnhancedKeyMixer = None  # type: ignore
