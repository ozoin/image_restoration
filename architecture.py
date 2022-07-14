import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


# class REDNet20(nn.Module):
#     def __init__(self, n_in_channels: int = 4,  num_layers: int = 10, num_features: int = 64, kernel_size: int = 6):
#         super(REDNet20, self).__init__()
#         self.num_layers = num_layers

#         conv_layers = []
#         deconv_layers = []

#         conv_layers.append(nn.Sequential(nn.Conv2d(n_in_channels, num_features, kernel_size=kernel_size, stride=2, padding=0),
#                                          nn.ReLU(inplace=True)))
#         for i in range(num_layers - 1):
#             conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=kernel_size, padding=0),
#                                              nn.ReLU(inplace=True)))

#         for i in range(num_layers - 1):
#             deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=kernel_size, padding=0),
#                                                nn.ReLU(inplace=True)))
#         deconv_layers.append(nn.ConvTranspose2d(
#             num_features, 3, kernel_size=kernel_size, stride=2, padding=0, output_padding=0))

#         self.conv_layers = nn.Sequential(*conv_layers)
#         self.deconv_layers = nn.Sequential(*deconv_layers)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x, concat):
#         residual = x
#         conv_feats = []
#         for i in range(self.num_layers):
#             x = self.conv_layers[i](x)
#             if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
#                 conv_feats.append(x)

#         conv_feats_idx = 0
#         for i in range(self.num_layers):
#             x = self.deconv_layers[i](x)
#             if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
#                 conv_feat = conv_feats[-(conv_feats_idx + 1)]
#                 conv_feats_idx += 1
#                 x = x + conv_feat
#                 x = self.relu(x)

#         residual = torch.concat((residual, concat), 1)
#         print(residual.shape)
#         x += residual
#         x = self.relu(x)

#         return x


# class REDNet10(nn.Module):
#     def __init__(self, n_in_channels=3, num_layers=5, num_features=64, kernel_size=6):
#         super(REDNet10, self).__init__()
#         conv_layers = []
#         deconv_layers = []

#         conv_layers.append(nn.Sequential(nn.Conv2d(n_in_channels, num_features, kernel_size=kernel_size, stride=2, padding=1),
#                                          nn.ReLU(inplace=True)))
#         for i in range(num_layers - 1):
#             conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=kernel_size, padding=1),
#                                              nn.ReLU(inplace=True)))

#         for i in range(num_layers - 1):
#             deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=kernel_size, padding=1),
#                                                nn.ReLU(inplace=True)))
#         deconv_layers.append(nn.ConvTranspose2d(
#             num_features, 3, kernel_size=kernel_size, stride=2, padding=1, output_padding=1))

#         self.conv_layers = nn.Sequential(*conv_layers)
#         self.deconv_layers = nn.Sequential(*deconv_layers)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x
#         out = self.conv_layers(x)
#         out = self.deconv_layers(out)
#         out += residual
#         return out


class AE(nn.Module):

    def __init__(self, neurons: int = 256,
                 kernel_size: int = 7):

        super(AE, self).__init__()
        self.neurons = neurons
        self.encoder = nn.Sequential(
            nn.Conv2d(4, self.neurons, kernel_size=kernel_size),
            nn.GELU(),
            nn.Conv2d(self.neurons, self.neurons, kernel_size=kernel_size),
            nn.GELU(),
            nn.Conv2d(self.neurons, self.neurons, kernel_size=kernel_size),
            nn.GELU(),
            nn.Conv2d(self.neurons, self.neurons, kernel_size=kernel_size),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(2, mode='nearest'),
            nn.ConvTranspose2d(self.neurons, self.neurons,
                               kernel_size=kernel_size),
            nn.GELU(),
            nn.ConvTranspose2d(self.neurons, self.neurons,
                               kernel_size=kernel_size),
            nn.GELU(),
            nn.ConvTranspose2d(self.neurons, self.neurons,
                               kernel_size=kernel_size),
            nn.GELU(),
            nn.ConvTranspose2d(self.neurons, 3, kernel_size=kernel_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AlterSimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 6, n_hidden_layers: int = 4, n_kernels: int = 64, kernel_size: int = 3):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()

        encoder, decoder = [], []
        self.kernels = n_kernels

        for _ in range(n_hidden_layers):
            encoder.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=self.kernels,
                kernel_size=kernel_size,
                padding=2
            ))
            encoder.append(torch.nn.ReLU())
            n_in_channels = self.kernels
            self.kernels = self.kernels * 2

        encoder.append(torch.nn.MaxPool2d(kernel_size-2, stride=1))
        self.enc_in_kernels = self.kernels // 2

        decoder.append(torch.nn.Upsample(scale_factor=1,
                                         mode='nearest'))
        for _ in range(n_hidden_layers - 1):
            decoder.append(torch.nn.ConvTranspose2d(
                in_channels=self.enc_in_kernels,
                out_channels=self.enc_in_kernels // 2,
                kernel_size=kernel_size,
                padding=2
            ))

            decoder.append(torch.nn.ReLU())
            self.enc_in_kernels = self.enc_in_kernels // 2

        decoder.append(torch.nn.ConvTranspose2d(
            in_channels=self.enc_in_kernels,
            out_channels=3,
            kernel_size=kernel_size,
        ))

        self.hidden_layers = torch.nn.Sequential(*encoder)
        self.decoder_layers = torch.nn.Sequential(*decoder)

    def _init_weights(self):
        pass

    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(
            x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        pred = self.decoder_layers(cnn_out)
        pred = torch.nn.functional.tanh(pred)
        return pred


class SimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 6, n_hidden_layers: int = 4, n_kernels: int = 64, kernel_size: int = 3):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()

        encoder, decoder = [], []
        self.kernels = n_kernels

        for _ in range(n_hidden_layers):
            encoder.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=self.kernels,
                kernel_size=kernel_size,
                padding=2
            ))
            encoder.append(torch.nn.ReLU())
            n_in_channels = self.kernels
            self.kernels = self.kernels * 2

        encoder.append(torch.nn.MaxPool2d(kernel_size-2, stride=1))
        self.enc_in_kernels = self.kernels // 2

        decoder.append(torch.nn.Upsample(scale_factor=1,
                                         mode='nearest'))
        for _ in range(n_hidden_layers - 1):
            decoder.append(torch.nn.ConvTranspose2d(
                in_channels=self.enc_in_kernels,
                out_channels=self.enc_in_kernels // 2,
                kernel_size=kernel_size,
                padding=2
            ))

            decoder.append(torch.nn.ReLU())
            self.enc_in_kernels = self.enc_in_kernels // 2

        decoder.append(torch.nn.ConvTranspose2d(
            in_channels=self.enc_in_kernels,
            out_channels=3,
            kernel_size=kernel_size,
        ))

        self.hidden_layers = torch.nn.Sequential(*encoder)
        self.decoder_layers = torch.nn.Sequential(*decoder)

    def _init_weights(self):
        pass

    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(
            x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        pred = self.decoder_layers(cnn_out)
        pred = torch.nn.functional.tanh(pred)
        return pred
