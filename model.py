import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # Thêm các lớp convolutional và activation khác cho encoder

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # Thực hiện các phép tính qua các lớp convolutional khác của encoder
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Thêm các lớp deconvolutional và activation cho decoder
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Thực hiện các phép tính qua các lớp deconvolutional và activation của decoder
        x = self.deconv1(x)
        x = self.sigmoid(x)
        return x


class InpaintingGAN(nn.Module):
    def __init__(self):
        super(InpaintingGAN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x
