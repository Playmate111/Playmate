import torch
from torch import nn
import torch.nn.functional as F
from .util import SPADEResnetBlock


class SPADEDecoder(nn.Module):
    def __init__(self, upscale=1, max_features=256, block_expansion=64, input_channels=256, out_channels=3, num_down_blocks=2):
        for i in range(num_down_blocks):
            input2_channels = min(max_features, block_expansion * (2 ** (i + 1)))
        self.upscale = upscale
        super().__init__()
        norm_G = 'spadespectralinstance'
        label_num_channels = input_channels  # 256
        inter_channels = 64

        self.fc = nn.Conv2d(input_channels, 2 * input2_channels, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * input2_channels, 2 * input2_channels, norm_G, label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(2 * input2_channels, 2 * input2_channels, norm_G, label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(2 * input2_channels, 2 * input2_channels, norm_G, label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(2 * input2_channels, 2 * input2_channels, norm_G, label_num_channels)
        self.G_middle_4 = SPADEResnetBlock(2 * input2_channels, 2 * input2_channels, norm_G, label_num_channels)
        self.G_middle_5 = SPADEResnetBlock(2 * input2_channels, 2 * input2_channels, norm_G, label_num_channels)
        self.up_0 = SPADEResnetBlock(2 * input2_channels, input2_channels, norm_G, label_num_channels)
        self.up_1 = SPADEResnetBlock(input2_channels, inter_channels, norm_G, label_num_channels)
        self.up = nn.Upsample(scale_factor=2)

        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(inter_channels, 3, 3, padding=1)
        else:
            if out_channels == 3:
                self.conv_img = nn.Sequential(
                    nn.Conv2d(inter_channels, out_channels * (2 * 2), kernel_size=3, padding=1),
                    nn.PixelShuffle(upscale_factor=2)
                )
                self.mask_conv_img = None
            elif out_channels == 4:
                self.conv_img = nn.Sequential(
                    nn.Conv2d(inter_channels, 3 * (2 * 2), kernel_size=3, padding=1),
                    nn.PixelShuffle(upscale_factor=2)
                )
                self.mask_conv_img = nn.Sequential(
                    nn.Conv2d(inter_channels, 1 * (2 * 2), kernel_size=3, padding=1),
                    nn.PixelShuffle(upscale_factor=2)
                )

    def forward(self, feature, z=None):
        seg = feature  # Bx256x64x64
        if z is not None:
            x = self.fc(z)  # Bx512x64x64
        else:
            x = self.fc(feature)  # Bx512x64x64
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)

        x = self.up(x)  # Bx512x64x64 -> Bx512x128x128
        x = self.up_0(x, seg)  # Bx512x128x128 -> Bx256x128x128
        x = self.up(x)  # Bx256x128x128 -> Bx256x256x256
        x = self.up_1(x, seg)  # Bx256x256x256 -> Bx64x256x256

        x = F.leaky_relu(x, 2e-1)
        pred_out = torch.sigmoid(self.conv_img(x))  # Bx64x256x256 -> Bx3xHxW
        if self.mask_conv_img is not None:
            pred_mask = torch.sigmoid(self.mask_conv_img(x))  # Bx64x256x256 -> Bx1xHxW
            pred_out = torch.cat([pred_out, pred_mask], dim=1)

        return pred_out
