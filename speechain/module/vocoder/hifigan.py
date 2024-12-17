import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels, channels, kernel_size, dilation=d, padding="same"
                    ),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels, channels, kernel_size, dilation=1, padding="same"
                    ),
                )
                for d in dilation
            ]
        )

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x


class MRF(nn.Module):
    def __init__(
        self,
        channels,
        kernel_sizes=(3, 7, 11),
        dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResBlock(channels, k, d) for k, d in zip(kernel_sizes, dilations)]
        )

    def forward(self, x):
        return sum([block(x) for block in self.resblocks])


class HiFiGAN(nn.Module):
    def __init__(
        self,
        in_channels=80,
        upsample_initial_channel=512,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
    ):
        super().__init__()

        self.pre_net = nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)

        ups_in_channels = [
            upsample_initial_channel // (2**i) for i in range(len(upsample_rates))
        ]
        ups_out_channels = [
            upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(upsample_rates))
        ]

        self.upsamples = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                nn.ConvTranspose1d(
                    ups_in_channels[i],
                    ups_out_channels[i],
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            self.mrfs.append(MRF(ups_out_channels[i]))

        self.post_net = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(ups_out_channels[-1], 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.pre_net(x)
        for up, mrf in zip(self.upsamples, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)
        x = self.post_net(x)
        return x.squeeze(1)

    def decode_batch(self, feats):
        """Convert mel-spectrograms to waveforms
        Args:
            feats: (batch, time, n_mels)
        Returns:
            waveforms: (batch, time')
        """
        self.eval()
        with torch.no_grad():
            x = feats.transpose(1, 2)  # (B, n_mels, T)
            return self.forward(x)

    @classmethod
    def from_hparams(cls, source, savedir=None, run_opts=None):
        """Load pretrained model
        Args:
            source: Model identifier (e.g. "speechbrain/tts-hifigan-ljspeech")
            savedir: Directory to save model weights
            run_opts: Runtime options including device
        Returns:
            model: Loaded HiFiGAN model
        """
        model = cls()

        # Set device
        if run_opts and "device" in run_opts:
            model = model.to(run_opts["device"])

        # Load pretrained weights based on source
        if savedir:
            weights_path = os.path.join(savedir, "generator.pth")
            if os.path.exists(weights_path):
                model.load_state_dict(
                    torch.load(weights_path, map_location=run_opts["device"])
                )

        return model
