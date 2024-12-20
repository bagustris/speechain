"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""

from typing import Dict

import torch

from speechain.module.abs import Module
from speechain.module.postnet.token import TokenPostnet
from speechain.utilbox.import_util import import_class
from speechain.utilbox.train_util import make_mask_from_len


class ARASRDecoder(Module):
    """"""

    def module_init(self, embedding: Dict, decoder: Dict, vocab_size: int = None):
        """

        Args:
            embedding:
            decoder:
            vocab_size:

        """
        # temporary register for connecting two sequential modules
        _prev_output_size = None

        # embedding layer of the E2E ASR decoder
        embedding_class = import_class("speechain.module." + embedding["type"])
        embedding["conf"] = (
            dict() if "conf" not in embedding.keys() else embedding["conf"]
        )
        self.embedding = embedding_class(vocab_size=vocab_size, **embedding["conf"])
        _prev_output_size = self.embedding.output_size

        # main body of the E2E ASR decoder
        decoder_class = import_class("speechain.module." + decoder["type"])
        decoder["conf"] = dict() if "conf" not in decoder.keys() else decoder["conf"]
        self.decoder = decoder_class(input_size=_prev_output_size, **decoder["conf"])
        _prev_output_size = self.decoder.output_size

        # token prediction layer for the E2E ASR decoder
        self.postnet = TokenPostnet(input_size=_prev_output_size, vocab_size=vocab_size)

    def forward(
        self,
        enc_feat: torch.Tensor,
        enc_feat_mask: torch.Tensor,
        text: torch.Tensor,
        text_len: torch.Tensor,
    ):
        """

        # Args:
            enc_feat:
            enc_feat_mask:
            text:
            text_len:

        Returns:

        """
        # Text Embedding
        emb_text = self.embedding(text)

        # mask generation for the input text
        text_mask = make_mask_from_len(text_len)
        if text.is_cuda:
            text_mask = text_mask.cuda(text.device)

        dec_feat, self_attmat, encdec_attmat, hidden = self.decoder(
            src=enc_feat, src_mask=enc_feat_mask, tgt=emb_text, tgt_mask=text_mask
        )
        return self.postnet(dec_feat), self_attmat, encdec_attmat, hidden
