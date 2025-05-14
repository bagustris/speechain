"""
Author: Heli Qi
Affiliation: NAIST
Date: 2022.09
"""

from typing import List

import torch

from speechain.module.abs import Module
from speechain.module.prenet.conv1d import Conv1dEv


class Conv1dPostnet(Module):
    """
    The Conv1d postnet for TTS. Usually used after the Transformer TTS decoder.
    This prenet is made up of only Conv1d blocks each of which contains:
     1. a Conv1d layer
     2. a BatchNorm1d layer
     3. an activation function
     4. a Dropout layer.

    Reference:
        Neural Speech Synthesis with Transformer Network
        https://ojs.aaai.org/index.php/AAAI/article/view/4642/4520
    """

    def module_init(
        self,
        feat_dim: int = None,
        conv_dims: int or List[int] = [512, 512, 512, 512, 0],
        conv_kernel: int = 5,
        conv_stride: int = 1,
        conv_padding_mode: str = "same",
        conv_batchnorm: bool = True,
        conv_activation: str = "Tanh",
        conv_dropout: float or List[float] = None,
        zero_centered: bool = False,
    ):
        """

        Args:
            feat_dim: int
                The dimension of input acoustic feature tensors.
                Used for calculating the in_features of the first Linear layer.
            conv_dims: List[int] or int
                The values of out_channels of each Conv1d layer.
                If a list of integers is given, multiple Conv1d layers will be initialized.
                If an integer is given, there will be only one Conv1d layer
                -1: same size as the previous convolutional layer's dim
                0: same size as the input feat_dim
            conv_kernel: int
                The value of kernel_size of all Conv1d layers.
            conv_stride: int
                The value of stride of all Conv1d layers.
            conv_padding_mode: str
                The padding mode of convolutional layers. Must be one of ['valid', 'full', 'same', 'causal'].
            conv_batchnorm: bool
                Whether a BatchNorm1d layer is added right after a Conv1d layer
            conv_activation: str
                The type of the activation function after all Conv1d layers.
                None means no activation function is needed.
            conv_dropout: float or List[float]
                The values of p rate of the Dropout layer after each Linear layer.
            zero_centered: bool
                Whether the output of this module is centered at 0.
                If the specified activation function changes the centroid of the output distribution, e.g. ReLU and
                LeakyReLU, the activation function won't be attached to the final Linear layer if zer_centered is set
                to True.

        """
        # Convolution arguments checking
        assert isinstance(
            conv_dims, (List, int)
        ), "The dimensions of convolutional layers must be given as a list of integers or an integer!"
        assert isinstance(
            conv_kernel, int
        ), "The sizes of convolutional kernels must be given as an integer!"
        assert isinstance(
            conv_stride, int
        ), "The lengths of convolutional strides must be given as an integer!"
        if conv_dropout is not None:
            assert isinstance(
                conv_dropout, (List, float)
            ), "The dropout rates of convolutional layers must be given as a list of integers or an integer!"

        # input_size initialization
        if self.input_size is not None:
            feat_dim = self.input_size
        else:
            assert feat_dim is not None

        # register convolution arguments
        self.conv_dims = conv_dims if isinstance(conv_dims, List) else [conv_dims]
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_padding_mode = conv_padding_mode
        self.conv_dropout = conv_dropout

        # Conv1d layers initialization
        _prev_dim = feat_dim
        _tmp_conv = []
        for i in range(len(self.conv_dims)):
            # 0 means go back to the input feat_dim
            if self.conv_dims[i] == 0:
                self.conv_dims[i] = feat_dim
            # -1 means equal to the previous layer
            elif self.conv_dims[i] == -1:
                self.conv_dims[i] = self.conv_dims[i - 1]
            _tmp_conv.append(
                # don't include bias in the convolutional layer if it is followed by a batchnorm layer
                # reference: https://stackoverflow.com/questions/46256747/can-not-use-both-bias-and-batch-normalization-in-convolution-layers
                Conv1dEv(
                    in_channels=_prev_dim,
                    out_channels=self.conv_dims[i],
                    kernel_size=self.conv_kernel,
                    stride=self.conv_stride,
                    padding_mode=self.conv_padding_mode,
                    bias=not conv_batchnorm,
                )
            )
            # BatchNorm is better to be placed before activation
            # reference: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
            if conv_batchnorm:
                _tmp_conv.append(torch.nn.BatchNorm1d(self.conv_dims[i]))
            if conv_activation is not None:
                # no 'ReLU'-series activation is added for the last layer if zero_centered is specified
                if i != len(self.conv_dims) - 1 or not (
                    zero_centered and "ReLU" in conv_activation
                ):
                    _tmp_conv.append(getattr(torch.nn, conv_activation)())
            if conv_dropout is not None:
                _tmp_conv.append(
                    torch.nn.Dropout(
                        p=(
                            self.conv_dropout
                            if not isinstance(self.conv_dropout, List)
                            else self.conv_dropout[i]
                        )
                    )
                )
            _prev_dim = conv_dims[i]
        self.conv = torch.nn.Sequential(*_tmp_conv)
        self.output_size = _prev_dim

    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        """

        Args:
            feat: (batch, feat_maxlen, feat_dim)
                The input feature tensors.
            feat_len: (batch,)
                The length of each feature tensor.

        Returns: feat, feat_len
            The embedded feature vectors with their lengths.

        """
        # forward the convolutional layers
        # (batch, feat_maxlen, feat_dim) -> (batch, feat_dim, feat_maxlen)
        feat = feat.transpose(1, 2)
        # (batch, feat_dim, feat_maxlen) -> (batch, conv_dim, feat_maxlen)
        feat = self.conv(feat)
        # (batch, conv_dim, feat_maxlen) -> (batch, feat_maxlen, conv_dim)
        return feat.transpose(1, 2)
