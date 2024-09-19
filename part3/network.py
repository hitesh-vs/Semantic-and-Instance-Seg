import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class UNetResNet18Scratch(nn.Module):
    def __init__(self, num_classes=1):
        super(UNetResNet18Scratch, self).__init__()
        # Encoder part (downsampling)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Input is RGB (3 channels)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.encoder2 = BasicBlock(64, 64)
        self.encoder3 = BasicBlock(64, 128, stride=2)
        self.encoder4 = BasicBlock(128, 256, stride=2)
        self.encoder5 = BasicBlock(256, 512, stride=2)

        # Decoder with transposed convolutions (upsampling)
        self.upconv4 = self.upconv(512, 256)
        self.upconv3 = self.upconv(256, 128)
        self.upconv2 = self.upconv(128, 64)
        self.upconv1 = self.upconv(64, 64)

        # Final convolution for binary output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder part
        e1 = self.encoder1(x)  # First block output (128x128)
        e2 = self.encoder2(e1)  # Second block output (64x64)
        e3 = self.encoder3(e2)  # Third block output (32x32)
        e4 = self.encoder4(e3)  # Fourth block output (16x16)
        e5 = self.encoder5(e4)  # Fifth block output (8x8)

        # Decoder part with skip connections
        d4 = self.upconv4(e5)  # Upsample (8x8 -> 16x16)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)  # Ensure size matches e4
        d4 = d4 + e4  # Skip connection with e4

        d3 = self.upconv3(d4)  # Upsample (16x16 -> 32x32)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)  # Ensure size matches e3
        d3 = d3 + e3  # Skip connection with e3

        d2 = self.upconv2(d3)  # Upsample (32x32 -> 64x64)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)  # Ensure size matches e2
        d2 = d2 + e2  # Skip connection with e2

        d1 = self.upconv1(d2)  # Upsample (64x64 -> 128x128)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)  # Ensure size matches e1
        d1 = d1 + e1  # Skip connection with e1

        # Final output layer (binary mask, 128x128 -> 256x256)
        out = self.final_conv(d1)
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)  # Ensure final output is 256x256
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class UNetMobileNetV2Scratch(nn.Module):
    def __init__(self, num_classes=1):
        super(UNetMobileNetV2Scratch, self).__init__()

        # Encoder layers (based on MobileNetV2 architecture)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 512, stride=2),
            DepthwiseSeparableConv(512, 1024, stride=2)
        )

        # Decoder layers with transposed convolutions and ReLU activations
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2), nn.ReLU(inplace=True)),
            nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.ReLU(inplace=True)),
            nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU(inplace=True)),
            nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU(inplace=True)),
            nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(inplace=True))
        ])

        # 1x1 convolutions for matching the number of channels in skip connections
        self.match_channels = nn.ModuleList([
            nn.Conv2d(1024, 512, kernel_size=1),  # Match encoder e5 with decoder d5
            nn.Conv2d(512, 256, kernel_size=1),   # Match encoder e4 with decoder d4
            nn.Conv2d(256, 128, kernel_size=1),   # Match encoder e3 with decoder d3
            nn.Conv2d(128, 64, kernel_size=1),    # Match encoder e2 with decoder d2
            nn.Conv2d(64, 32, kernel_size=1)      # Match encoder e1 with decoder d1
        ])

        # Final classification layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        # Reverse the order of skips for decoder use
        skips = skips[::-1]

        # Apply decoder layers and use skip connections with 1x1 convolutions for channel matching
        for i, (d_layer, match_layer) in enumerate(zip(self.decoder, self.match_channels)):
            x = d_layer(x)
            skip = match_layer(skips[i])
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip  # Skip connection

        # Final output layer
        out = self.final_conv(x)

        # Upsample the output to match the input size (e.g., 256x256)
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)

        return out
