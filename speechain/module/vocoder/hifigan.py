"""
HiFi-GAN Vocoder Module

This module provides a HiFi-GAN vocoder implementation compatible with
SpeechBrain's pretrained weights from HuggingFace Hub.

The implementation uses weight normalization as in the SpeechBrain checkpoint.

Authors:
    Speechain Authors
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


# Default HiFi-GAN configuration (LJSpeech)
HIFIGAN_DEFAULT_CONFIG = {
    "in_channels": 80,
    "out_channels": 1,
    "resblock_type": "1",
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "resblock_kernel_sizes": [3, 7, 11],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "upsample_initial_channel": 512,
    "upsample_factors": [8, 8, 2, 2],
}


def get_padding(kernel_size, dilation=1):
    """Calculate padding for a convolution layer."""
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    """Residual Block Type 1 for HiFi-GAN."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[i],
                        padding=get_padding(kernel_size, dilation[i]),
                    )
                )
                for i in range(len(dilation))
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    """Residual Block Type 2 for HiFi-GAN."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[i],
                        padding=get_padding(kernel_size, dilation[i]),
                    )
                )
                for i in range(len(dilation))
            ]
        )

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class HiFiGAN(nn.Module):
    """
    HiFi-GAN Generator Module.

    This implementation is compatible with SpeechBrain's pretrained weights.
    It uses weight normalization and the same architecture as the SpeechBrain model.

    Args:
        in_channels: Number of input channels (mel spectrogram bins)
        out_channels: Number of output channels (1 for mono audio)
        resblock_type: Type of residual block ("1" or "2")
        resblock_dilation_sizes: Dilation sizes for residual blocks
        resblock_kernel_sizes: Kernel sizes for residual blocks
        upsample_kernel_sizes: Kernel sizes for upsampling layers
        upsample_initial_channel: Initial number of channels after first conv
        upsample_factors: Upsampling factors for each layer
    """

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        resblock_type="1",
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes=[3, 7, 11],
        upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=512,
        upsample_factors=[8, 8, 2, 2],
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)

        # Initial convolution - named 'conv_pre' to match SpeechBrain
        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        )

        # Select residual block type
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        # Upsampling layers - named 'ups' to match SpeechBrain
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # Residual blocks - named 'resblocks' to match SpeechBrain
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        # Final convolution - named 'conv_post' to match SpeechBrain
        self.conv_post = weight_norm(nn.Conv1d(ch, out_channels, 7, 1, padding=3))

    def forward(self, x):
        """
        Forward pass of HiFi-GAN generator.

        Args:
            x: Input mel spectrogram tensor of shape (batch, mel_channels, time)

        Returns:
            Audio waveform tensor of shape (batch, 1, time * product(upsample_factors))
        """
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""

        for up_layer in self.ups:
            remove_weight_norm(up_layer)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    @classmethod
    def from_pretrained(
        cls,
        model_name="speechbrain/tts-hifigan-ljspeech",
        cache_dir=None,
        device=None,
    ):
        """
        Load pretrained HiFi-GAN from HuggingFace Hub.

        Args:
            model_name: HuggingFace model name or path
            cache_dir: Directory to cache downloaded files
            device: Device to load model onto

        Returns:
            Loaded HiFiGAN model
        """
        from huggingface_hub import hf_hub_download

        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/speechain/vocoders")

        os.makedirs(cache_dir, exist_ok=True)

        # Download checkpoint
        ckpt_path = hf_hub_download(
            repo_id=model_name,
            filename="generator.ckpt",
            cache_dir=cache_dir,
        )

        print(f"Loading HiFi-GAN from {ckpt_path}")

        # Create model with default config
        model = cls(**HIFIGAN_DEFAULT_CONFIG)

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "generator" in checkpoint:
            state_dict = checkpoint["generator"]
        else:
            state_dict = checkpoint

        # Remap keys from SpeechBrain format to our format
        # SpeechBrain uses: conv_pre.conv.weight_g -> conv_pre.weight_g
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove the extra .conv. part from SpeechBrain keys
            new_key = key.replace(".conv.", ".")
            new_state_dict[new_key] = value

        # Try to load with remapped keys
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("Loaded HiFi-GAN checkpoint successfully")
        except RuntimeError as e:
            print(f"Warning: Strict loading failed: {e}")
            # Try with strict=False as fallback
            model.load_state_dict(new_state_dict, strict=False)
            print("Loaded HiFi-GAN checkpoint with strict=False")

        if device is not None:
            model = model.to(device)

        model.eval()
        return model

    def decode_batch(self, mel):
        """
        Decode mel spectrogram to waveform.

        This method is compatible with SpeechBrain's interface.

        Args:
            mel: Mel spectrogram tensor of shape (batch, time, mel_channels)
                 Note: SpeechBrain uses (batch, time, channels) format

        Returns:
            Audio waveform tensor
        """
        # Transpose from (batch, time, channels) to (batch, channels, time)
        if mel.dim() == 3 and mel.size(-1) == self.in_channels:
            mel = mel.transpose(1, 2)

        with torch.no_grad():
            audio = self.forward(mel)

        return audio.squeeze(1)  # Remove channel dimension


def load_hifigan_vocoder(
    checkpoint_path=None,
    model_name="speechbrain/tts-hifigan-ljspeech",
    device=None,
):
    """
    Load HiFi-GAN vocoder.

    This function provides a simple interface to load the vocoder
    either from a local checkpoint or from HuggingFace Hub.

    Args:
        checkpoint_path: Path to local checkpoint (optional)
        model_name: HuggingFace model name for download
        device: Device to load model onto

    Returns:
        Loaded HiFiGAN model
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading HiFi-GAN from local path: {checkpoint_path}")
        model = HiFiGAN(**HIFIGAN_DEFAULT_CONFIG)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "generator" in checkpoint:
            state_dict = checkpoint["generator"]
        else:
            state_dict = checkpoint

        # Remap keys from SpeechBrain format
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace(".conv.", ".")
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)

        if device is not None:
            model = model.to(device)

        model.eval()
        return model
    else:
        return HiFiGAN.from_pretrained(model_name, device=device)


if __name__ == "__main__":
    # Test loading
    print("Testing HiFi-GAN loading...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HiFiGAN.from_pretrained(device=device)
    print(f"Model loaded on {device}")

    # Test forward pass
    mel = torch.randn(1, 80, 100).to(device)
    with torch.no_grad():
        audio = model(mel)
    print(f"Input mel shape: {mel.shape}")
    print(f"Output audio shape: {audio.shape}")
    print("Test passed!")
