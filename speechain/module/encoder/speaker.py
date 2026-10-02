"""Speaker embedding models (ECAPA-TDNN & X-vector) with pretrained weights.

This module contains faithful PyTorch ports of the ECAPA-TDNN and X-vector
speaker embedding models, the Fbank acoustic frontend, and the input
normalization from the SpeechBrain toolkit
(https://github.com/speechbrain/speechbrain, Apache-2.0 license; original
authors: Hwidong Na (ECAPA-TDNN), Mirco Ravanelli (X-vector, features)).
The port is written so that the pretrained checkpoints published on the
HuggingFace hub (``speechbrain/spkrec-ecapa-voxceleb`` and
``speechbrain/spkrec-xvect-voxceleb``) load with an exact key match
(``strict=True``) and reproduce the reference embeddings numerically, without
requiring the SpeechBrain package itself (see issue #4).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _length_to_mask(length, max_len, device):
    """Boolean mask of shape (batch, max_len), True for valid positions."""
    return torch.arange(max_len, device=device).unsqueeze(0) < length.unsqueeze(1)


def _make_padding_mask(x, lengths, length_dim=1, eps=1e-8):
    """Boolean padding mask like SpeechBrain's make_padding_mask.

    True for valid positions. The returned mask broadcasts over the
    non-(batch, length) dimensions of x.
    """
    if lengths is None:
        lengths = torch.ones(x.size(0), device=x.device)
    max_len = x.size(length_dim)
    abs_lengths = (lengths * max_len - eps).unsqueeze(1)
    mask = torch.arange(max_len, device=x.device).unsqueeze(0) < abs_lengths
    for dim in range(1, x.ndim):
        if dim != length_dim:
            mask = mask.unsqueeze(dim)
    return mask


class Conv1d(nn.Module):
    """1D convolution operating on (batch, channels, time) tensors.

    Equivalent to SpeechBrain's Conv1d with skip_transpose=True and its
    default 'same' padding in 'reflect' mode (this padding choice is
    numerically verified against the reference implementation in
    speechain/module/vocoder/hifigan.py).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
            groups=groups,
            padding_mode="reflect",
        )

    def forward(self, x):
        return self.conv(x)


class BatchNorm1d(nn.Module):
    """Batch normalization operating on (batch, channels, time) tensors.

    Equivalent to SpeechBrain's BatchNorm1d with skip_transpose=True.
    """

    def __init__(self, input_size):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        return self.norm(x)


class Linear(nn.Module):
    """Linear layer equivalent to SpeechBrain's Linear wrapper."""

    def __init__(self, input_size, n_neurons, bias=True):
        super().__init__()
        self.w = nn.Linear(input_size, n_neurons, bias=bias)

    def forward(self, x):
        return self.w(x)


class TDNNBlock(nn.Module):
    """An implementation of TDNN (ported from SpeechBrain's ECAPA_TDNN.py)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
        groups=1,
        dropout=0.0,
    ):
        super().__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        return self.dropout(self.norm(self.activation(self.conv(x))))


class Res2NetBlock(nn.Module):
    """An implementation of Res2NetBlock w/ dilation (ported from SpeechBrain)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        scale=8,
        kernel_size=3,
        dilation=1,
        dropout=0.0,
    ):
        super().__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for _ in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        return torch.cat(y, dim=1)


class SEBlock(nn.Module):
    """An implementation of squeeze-and-excitation block (ported from SpeechBrain)."""

    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = _length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling (ported from SpeechBrain's ECAPA_TDNN.py).

    Returns the concatenated attention-weighted mean and std of the input.
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """x: (batch, channels, time); lengths: relative lengths or None."""
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = _length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance
        if self.global_context:
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SERes2NetBlock(nn.Module):
    """TDNN-Res2Net-TDNN-SEBlock building block of ECAPA-TDNN (ported)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        activation=torch.nn.ReLU,
        groups=1,
        dropout=0.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
            dropout=dropout,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
            dropout=dropout,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual


class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN speaker embedding model (ported from SpeechBrain).

    'ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification' (https://arxiv.org/abs/2005.07143).

    The default architecture matches the pretrained
    ``speechbrain/spkrec-ecapa-voxceleb`` checkpoint
    (channels=[1024, 1024, 1024, 1024, 3072]).
    """

    def __init__(
        self,
        input_size,
        lin_neurons=192,
        activation=torch.nn.ReLU,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        groups=[1, 1, 1, 1, 1],
        dropout=0.0,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
                dropout,
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                    dropout=dropout,
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-2] * (len(channels) - 2),
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
            dropout=dropout,
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x, lengths=None):
        """x: (batch, time, channel). Returns (batch, 1, lin_neurons)."""
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)

        xl = []
        for layer in self.blocks:
            if isinstance(layer, TDNNBlock):
                x = layer(x)
            else:
                x = layer(x, lengths=lengths)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        x = x.transpose(1, 2)
        return x


class StatisticsPooling(nn.Module):
    """Statistics pooling layer for the X-vector model (ported from SpeechBrain).

    Returns the concatenated mean and std of the input along the time axis.
    Following the reference implementation, a tiny deterministic-scale
    perturbation is added to the statistics (see the forward function).
    """

    def __init__(self, return_mean=True, return_std=True):
        super().__init__()
        self.eps = 1e-5
        self.return_mean = return_mean
        self.return_std = return_std
        if not (self.return_mean or self.return_std):
            raise ValueError(
                "both of statistics are equal to False \n"
                "consider enabling mean and/or std statistic pooling"
            )

    def forward(self, x, lengths=None):
        """x: (batch, time, channels). Returns (batch, 1, 2 * channels)."""
        if lengths is None:
            if self.return_mean:
                mean = x.mean(dim=1)
            if self.return_std:
                std = x.std(dim=1)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                # Avoiding padded time steps
                actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))

                # computing statistics
                if self.return_mean:
                    mean.append(torch.mean(x[snt_id, 0:actual_size, ...], dim=0))
                if self.return_std:
                    std.append(torch.std(x[snt_id, 0:actual_size, ...], dim=0))
            if self.return_mean:
                mean = torch.stack(mean)
            if self.return_std:
                std = torch.stack(std)

        if self.return_mean:
            gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
            mean += gnoise
        if self.return_std:
            std = std + self.eps

        # Append mean and std of the batch
        if self.return_mean and self.return_std:
            pooled_stats = torch.cat((mean, std), dim=1)
            pooled_stats = pooled_stats.unsqueeze(1)
        elif self.return_mean:
            pooled_stats = mean.unsqueeze(1)
        else:
            pooled_stats = std.unsqueeze(1)

        return pooled_stats

    def _get_gauss_noise(self, shape_of_tensor, device="cpu"):
        """Returns a tensor of epsilon Gaussian noise (as the reference does)."""
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)
        return gnoise


class Xvector(nn.Module):
    """X-vector speaker embedding model (ported from SpeechBrain's Xvector.py).

    The default architecture matches the pretrained
    ``speechbrain/spkrec-xvect-voxceleb`` checkpoint
    (in_channels=24, tdnn_channels=[512, 512, 512, 512, 1500]).
    """

    def __init__(
        self,
        in_channels=24,
        activation=torch.nn.LeakyReLU,
        tdnn_blocks=5,
        tdnn_channels=[512, 512, 512, 512, 1500],
        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
        tdnn_dilations=[1, 2, 3, 1, 1],
        lin_neurons=512,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        # TDNN layers
        for block_index in range(tdnn_blocks):
            out_channels = tdnn_channels[block_index]
            self.blocks.extend(
                [
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=tdnn_kernel_sizes[block_index],
                        dilation=tdnn_dilations[block_index],
                    ),
                    activation(),
                    BatchNorm1d(input_size=out_channels),
                ]
            )
            in_channels = tdnn_channels[block_index]

        # Statistical pooling
        self.blocks.append(StatisticsPooling())

        # Final linear transformation
        self.blocks.append(
            Linear(
                input_size=out_channels * 2,
                n_neurons=lin_neurons,
                bias=True,
            )
        )

    def forward(self, x, lens=None):
        """x: (batch, time, channels). Returns (batch, 1, lin_neurons)."""
        # conv layers operate on (batch, channels, time)
        x = x.transpose(1, 2)
        for layer in self.blocks:
            if isinstance(layer, StatisticsPooling):
                # pooling operates on (batch, time, channels)
                x = layer(x.transpose(1, 2), lengths=lens)
            elif isinstance(layer, Linear):
                # Linear operates on the last dimension of (batch, 1, 2*channels)
                x = layer(x)
            else:
                # Conv1d / activation / BatchNorm1d operate on (batch, channels, time)
                x = layer(x)
        return x


class FbankFrontend(nn.Module):
    """Fbank acoustic frontend (ported from SpeechBrain's lobes/features.py).

    Computes log-mel filterbank features from raw waveforms exactly as the
    reference implementation does: STFT (center=True, constant padding,
    periodic Hann window) -> power spectrogram -> triangular mel filters in
    the HTK mel scale -> 10 * log10 -> per-utterance top_db clamping.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=400,
        n_mels=80,
        f_min=0.0,
        f_max=None,
        win_length=400,
        hop_length=160,
        amin=1e-10,
        top_db=80.0,
    ):
        super().__init__()
        if f_max is None:
            f_max = sample_rate // 2
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.amin = amin
        self.top_db = top_db

        self.register_buffer("window", torch.hann_window(win_length))
        self.register_buffer(
            "filters",
            self._triangular_filters(sample_rate, n_fft, n_mels, f_min, f_max),
        )

    @staticmethod
    def _to_mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    @classmethod
    def _triangular_filters(cls, sample_rate, n_fft, n_mels, f_min, f_max):
        """Triangular mel filters exactly as SpeechBrain's Filterbank."""
        mel = torch.linspace(cls._to_mel(f_min), cls._to_mel(f_max), n_mels + 2)
        hz = cls._to_hz(mel)

        # Computation of the filter bands
        band = hz[1:] - hz[:-1]
        band = band[:-1]
        f_central = hz[1:-1]

        # Frequency axis
        n_stft = n_fft // 2 + 1
        all_freqs = torch.linspace(0, sample_rate // 2, n_stft)

        # slope of each filter on the frequency grid
        slope = (all_freqs.unsqueeze(1) - f_central.unsqueeze(0)) / band.unsqueeze(0)
        left_side = slope + 1.0
        right_side = -slope + 1.0
        return torch.clamp(torch.min(left_side, right_side), min=0.0)

    def forward(self, wav):
        """wav: (batch, time). Returns (batch, time, n_mels) log-fbanks."""
        stft = torch.stft(
            wav,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window.to(wav.device),
            True,
            "constant",
            False,
            True,
            return_complex=True,
        )
        # power spectrogram (batch, freq, time) -> (batch, time, freq)
        mag = stft.real.pow(2) + stft.imag.pow(2)
        mag = mag.transpose(1, 2)

        fbanks = torch.matmul(mag, self.filters.to(wav.device))

        # 10 * log10 with per-utterance top_db clamping
        fbanks = 10 * torch.log10(torch.clamp(fbanks, min=self.amin))
        db_max = fbanks.amax(dim=(-2, -1), keepdim=True) - self.top_db
        return torch.maximum(fbanks, db_max)


class SentenceMeanNorm(nn.Module):
    """Per-utterance mean normalization over the time axis.

    Equivalent to SpeechBrain's InputNormalization with norm_type='sentence'
    and std_norm=False (as used by the pretrained speaker embedding models).
    """

    def forward(self, x, lengths=None):
        """x: (batch, time, channels). Returns the mean-normalized tensor."""
        mask = _make_padding_mask(x, lengths, length_dim=1)
        n = mask.sum(dim=1, keepdim=True)
        mean = (x * mask).sum(dim=1, keepdim=True) / n
        return x - mean


class EncoderClassifier(nn.Module):
    """Speaker embedding extractor with pretrained ECAPA-TDNN / X-vector models.

    Drop-in replacement for SpeechBrain's EncoderClassifier limited to the two
    pretrained models used by SpeeChain's TTS data pipeline:
    ``speechbrain/spkrec-ecapa-voxceleb`` and ``speechbrain/spkrec-xvect-voxceleb``.
    The checkpoints are downloaded from the HuggingFace hub on the first call
    and loaded into the ported models above with an exact key match.
    """

    def __init__(self, model_type="ecapa"):
        super().__init__()
        self.model_type = model_type
        if model_type == "ecapa":
            self.compute_features = FbankFrontend(n_mels=80)
            self.embedding_model = ECAPA_TDNN(input_size=80)
        elif model_type == "xvector":
            self.compute_features = FbankFrontend(n_mels=24)
            self.embedding_model = Xvector(in_channels=24)
        else:
            raise ValueError(
                f"Unknown speaker embedding model type: {model_type}. "
                f"It should be either 'ecapa' or 'xvector'."
            )
        self.mean_var_norm = SentenceMeanNorm()
        # global embedding statistics (loaded only when normalize=True is used)
        self.register_buffer("glob_mean", None)
        self.register_buffer("glob_std", None)

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        """Encodes the input audio into a single vector embedding.

        Args:
            wavs: Batch of waveforms of shape (batch, time) at 16 kHz.
            wav_lens: Relative lengths of the waveforms (batch,), where the
                longest one is 1.0. Used for ignoring the padding part.
            normalize: If True, the embeddings are normalized with the global
                statistics published together with the pretrained checkpoints
                (mean_var_norm_emb.ckpt).

        Returns:
            torch.Tensor: The speaker embeddings of shape (batch, 1, emb_dim).
        """
        if wavs.dim() == 1:
            wavs = wavs.unsqueeze(0)
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=wavs.device)

        self.eval()
        with torch.no_grad():
            wavs = wavs.float()
            feats = self.compute_features(wavs)
            feats = self.mean_var_norm(feats, wav_lens)
            embeddings = self.embedding_model(feats, wav_lens)
            if normalize:
                if self.glob_mean is None:
                    raise RuntimeError(
                        "Global embedding statistics are not loaded; "
                        "load the model with from_hparams() first."
                    )
                embeddings = (embeddings - self.glob_mean) / self.glob_std.clamp(
                    min=1e-8
                )
            # L2-normalize so averaging embeddings across utterances (spk_util.py) weighs
            # them equally regardless of magnitude, and so existing pretrained multi-speaker
            # TTS checkpoints (trained against unit-norm speaker conditioning vectors) keep
            # seeing the same input distribution
            return F.normalize(embeddings, p=2, dim=-1)

    @classmethod
    def from_hparams(cls, source, savedir=None, run_opts=None):
        """Load a pretrained speaker embedding model from the HuggingFace hub.

        Args:
            source: The HuggingFace repository id of the pretrained model
                (e.g., 'speechbrain/spkrec-ecapa-voxceleb').
            savedir: The directory where the downloaded checkpoints are cached.
            run_opts: A dict of run options; only the 'device' entry is used.

        Returns:
            EncoderClassifier: The pretrained model in eval() mode.
        """
        from huggingface_hub import hf_hub_download

        model = cls(model_type="ecapa" if "ecapa" in source else "xvector")

        # download the embedding model checkpoint if needed
        ckpt_path = hf_hub_download(
            repo_id=source, filename="embedding_model.ckpt", cache_dir=savedir
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # raise an error on any key mismatch instead of silently using
        # randomly-initialized weights
        model.embedding_model.load_state_dict(state_dict, strict=True)

        # the global embedding statistics are optional (only used by normalize=True)
        try:
            norm_path = hf_hub_download(
                repo_id=source,
                filename="mean_var_norm_emb.ckpt",
                cache_dir=savedir,
            )
        except Exception:
            norm_path = None
        if norm_path is not None:
            stats = torch.load(norm_path, map_location="cpu", weights_only=True)
            # keep the embedding dimension: (emb_dim,) -> (1, 1, emb_dim)
            model.glob_mean = stats["glob_mean"].reshape(1, 1, -1)
            model.glob_std = stats["glob_std"].reshape(1, 1, -1)

        device = run_opts.get("device", "cpu") if run_opts else "cpu"
        model = model.to(device)
        model.eval()
        return model
