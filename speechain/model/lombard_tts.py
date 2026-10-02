"""
Author: Bagus Tris Atmaja (with Claude Code)
Affiliation: NAIST
Date: 2026.09

Dynamically adaptive Lombard TTS in a machine speech chain
(S. Novitasari, S. Sakti, S. Nakamura, "A Machine Speech Chain Approach for
Dynamically Adaptive Lombard TTS in Static and Dynamic Noise Environments",
IEEE/ACM TASLP 2022; Interspeech 2021).

The TTS (FastSpeech2 with its variance adaptor for pitch, energy and duration)
receives auditory feedback about how its speech sounds in the noisy
environment:

    1. Z_SNR: the SNR embedding of the noisy speech given by an SNR predictor
       (power measurement).
    2. Z_ASR: the embedding of the ASR loss of the noisy speech given by a
       frozen ASR (speech intelligibility measurement).

Both embeddings are added to the TTS encoder output (h^e = h_trm + Z_SPK +
Z_SNR + Z_ASR). During inference, the machine speech chain runs in a closed
loop: speak -> listen (add noise, measure SNR & ASR loss) -> speak again,
until the ASR loss converges.

Training uses normal speech and synthetic Lombard speech of several noise
conditions (see speechain/datasets/pyscripts/lombard_synth.py). For a target
utterance of condition c, the feedback is obtained by listening to the
normal (reference) speech in the noise of condition c.
"""

import copy
import os
import warnings
from typing import Any, Dict, List

import torch

from speechain.inference import build_model_from_exp, load_checkpoint, load_exp_cfg
from speechain.model.nar_tts import FastSpeech2
from speechain.module.augment.noise_mixer import NoiseMixer, SNRSpec
from speechain.module.prenet.feedback_embed import FeedbackEmbedPrenet
from speechain.module.standalone.snr_predictor import SNRPredictor
from speechain.utilbox.audio_util import get_cached_resampler
from speechain.utilbox.data_loading_util import parse_path_args
from speechain.utilbox.tensor_util import to_cpu
from speechain.utilbox.train_util import make_mask_from_len, text2tensor_and_len


class LombardFastSpeech2(FastSpeech2):
    """FastSpeech2 with SNR and ASR-loss auditory feedback (Lombard machine speech
    chain)."""

    def module_init(
        self,
        snr_conditions: Dict[str, SNRSpec],
        snr_predictor: Dict,
        asr_exp_path: str = None,
        asr_test_model: str = "10_valid_accuracy_average",
        asr_model: Any = None,
        feedback: Dict = None,
        noise: Dict = None,
        feedback_sample_rate: int = 16000,
        train_noise_types: List[str] = None,
        snr_emb_detach: bool = True,
        **fs2_conf,
    ):
        """
        Args:
            snr_conditions: Dict[str, SNRSpec]
                The noise conditions used for training, mapping the condition name (the `snr_cond`
                of the dataset) to its SNR in dB, e.g. {'clean': null, 'snr0': 0, 'snr-10': -10}.
                The keys are also the SNR classes of the SNR predictor.
            snr_predictor: Dict
                The configuration of the SNRPredictor module (frontend, conv_dims, emb_dim, ...).
            asr_exp_path: str
                The experiment folder of the trained ASR used as the listener.
            asr_test_model: str
                The checkpoint name of the ASR.
            asr_model: Any
                An already-built ASR model (mainly for testing). Overrides asr_exp_path.
            feedback: Dict
                The configuration of the FeedbackEmbedPrenet (snr_coeff, asr_coeff, ...).
            noise: Dict
                The configuration of the NoiseMixer (noise_files, snr_jitter).
            feedback_sample_rate: int
                The sampling rate at which the noise is added and the ASR/SNR predictor listen.
            train_noise_types: List[str]
                The noise types randomly picked for each training utterance. Default: all noise
                types registered in the NoiseMixer.
            snr_emb_detach: bool
                Whether Z_SNR is detached before entering the TTS (the SNR predictor is then trained
                only by its classification loss).
            **fs2_conf:
                The arguments of FastSpeech2.module_init().
        """
        super().module_init(**fs2_conf)
        d_model = self.encoder.output_size

        self.snr_conditions = dict(snr_conditions)
        self.snr_classes = list(self.snr_conditions.keys())
        snr_predictor = copy.deepcopy(snr_predictor)
        snr_predictor.setdefault("emb_dim", d_model)
        self.snr_predictor = SNRPredictor(snr_classes=self.snr_classes, **snr_predictor)
        self.feedback_embed = FeedbackEmbedPrenet(
            d_model=d_model,
            snr_emb_dim=self.snr_predictor.emb_dim,
            **(feedback or dict()),
        )
        self.noise_mixer = NoiseMixer(
            sample_rate=feedback_sample_rate, **(noise or dict())
        )
        self.feedback_sample_rate = feedback_sample_rate
        self.train_noise_types = (
            self.noise_mixer.noise_types
            if train_noise_types is None
            else train_noise_types
        )
        self.snr_emb_detach = snr_emb_detach

        # the ASR listener is frozen and kept outside the module tree (like the LM of ARASR) so that
        # it is neither trained, re-initialized, nor saved in the checkpoints of this model
        self.asr_exp_path = asr_exp_path
        self.asr_test_model = asr_test_model
        self._asr_holder = [asr_model]
        self.resampler_cache = {}

    # --- Frozen ASR listener --- #
    @property
    def asr(self):
        """Lazily build the frozen ASR on the device of the TTS parameters."""
        device = next(self.encoder.parameters()).device
        if self._asr_holder[0] is None:
            assert self.asr_exp_path is not None, (
                "Please give asr_exp_path (the experiment folder of the ASR listener) in "
                "model['customize_conf'] of LombardFastSpeech2!"
            )
            exp_path = parse_path_args(self.asr_exp_path)
            exp_cfg = load_exp_cfg(exp_path)
            asr = build_model_from_exp(exp_cfg, exp_path, device)
            load_checkpoint(
                asr, exp_path, self.asr_test_model, device, trust_checkpoint=True
            )
            self._asr_holder[0] = asr
        asr = self._asr_holder[0]
        asr.eval()
        for para in asr.parameters():
            para.requires_grad = False
        if next(asr.parameters()).device != device:
            asr.to(device)
            asr.device = device
        return asr

    def train(self, mode: bool = True):
        super().train(mode)
        if self._asr_holder[0] is not None:
            self._asr_holder[0].eval()
        return self

    # --- Auditory feedback --- #
    def to_feedback_sr(self, wav: torch.Tensor, wav_len: torch.Tensor, orig_sr: int):
        """Resample waveforms (batch, wav_maxlen[, 1]) to the feedback sampling rate."""
        wav = wav.squeeze(-1) if wav.dim() == 3 else wav
        if orig_sr == self.feedback_sample_rate:
            return wav.float(), wav_len
        resampler = get_cached_resampler(
            self.resampler_cache, orig_sr, self.feedback_sample_rate, device=wav.device
        )
        # the resampler is Conv1d-based, so under AMP it would otherwise be autocast to fp16
        # even though its fp32 weights never change; keep the DSP path (and its output) in fp32
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav = resampler(wav.float())
        wav_len = (wav_len.float() * self.feedback_sample_rate / orig_sr).ceil().long()
        wav_len = wav_len.clamp(max=wav.size(1))
        # explicit max_len: wav_len.max() can fall short of wav.size(1) by a sample or two due to
        # resampling rounding, which would otherwise make the mask narrower than wav itself
        mask = make_mask_from_len(wav_len, max_len=wav.size(1), return_3d=False)
        wav = wav * mask.to(wav.device)
        return wav, wav_len

    @torch.no_grad()
    def asr_loss(self, wav: torch.Tensor, wav_len: torch.Tensor, text_asr: List[str]):
        """Per-utterance cross-entropy loss of the frozen ASR on noisy waveforms
        (batch, wav_maxlen) at the feedback sampling rate."""
        asr = self.asr
        # text2tensor_and_len() modifies the given list in place, so a copy is given
        text, text_len = text2tensor_and_len(
            text_list=list(text_asr),
            text2tensor_func=asr.tokenizer.text2tensor,
            ignore_idx=asr.tokenizer.ignore_idx,
        )
        text, text_len = text.to(wav.device), text_len.to(wav.device)
        # the frozen ASR's frontend assumes fp32 audio; don't trust the caller under AMP
        # (autocast doesn't retroactively upcast an already-fp16 tensor, it only affects new ops)
        with torch.autocast(device_type=wav.device.type, enabled=False):
            # the ASR forward removes <eos> from the input text; the targets are the shifted tokens
            outputs = asr.module_forward(
                feat=wav.float().unsqueeze(-1),
                feat_len=wav_len.clone(),
                text=text.clone(),
                text_len=text_len.clone(),
            )
        logits = outputs["logits"].float()
        target = text[:, 1 : logits.size(1) + 1].clone()
        tgt_mask = make_mask_from_len(text_len - 1, return_3d=False).to(wav.device)
        target[~tgt_mask] = asr.tokenizer.ignore_idx
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2),
            target,
            ignore_index=asr.tokenizer.ignore_idx,
            reduction="none",
        )
        loss = loss.sum(dim=-1) / tgt_mask.sum(dim=-1).clamp(min=1)
        # degenerate (e.g. extremely short) speech may give non-finite losses: treat it as unintelligible
        return torch.nan_to_num(loss, nan=1e4, posinf=1e4)

    def listen(
        self,
        wav: torch.Tensor,
        wav_len: torch.Tensor,
        orig_sr: int,
        snr: List[SNRSpec],
        noise_type: List[str] or str,
        text_asr: List[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Listen to the speech in the noisy environment: add noise, predict the SNR and calculate
        the ASR loss.

        Returns:
            Dict with noisy_wav, noisy_wav_len (at the feedback sampling rate), snr_applied,
            snr_logits, snr_emb, snr_frame_emb, snr_frame_len, asr_loss (None if text_asr is None).
        """
        wav, wav_len = self.to_feedback_sr(wav, wav_len, orig_sr)
        # the listeners need at least a few frames: pad extremely short speech with silence
        min_len = int(0.1 * self.feedback_sample_rate)
        if wav.size(1) < min_len:
            wav = torch.nn.functional.pad(wav, (0, min_len - wav.size(1)))
        wav_len = wav_len.clamp(min=min(min_len, wav.size(1)))
        with torch.no_grad():
            noisy, noisy_len, snr_applied = self.noise_mixer(
                wav, wav_len, snr, noise_type
            )
        snr_out = self.snr_predictor(noisy.unsqueeze(-1), noisy_len.clone())
        outputs = dict(
            noisy_wav=noisy,
            noisy_wav_len=noisy_len,
            snr_applied=snr_applied,
            snr_logits=snr_out["logits"],
            snr_emb=snr_out["emb"],
            snr_frame_emb=snr_out["frame_emb"],
            snr_frame_len=snr_out["feat_len"],
            asr_loss=None,
        )
        if text_asr is not None:
            outputs["asr_loss"] = self.asr_loss(noisy, noisy_len.clone(), text_asr)
        return outputs

    def snr_spec_of_cond(self, snr_cond: List[str]) -> List[SNRSpec]:
        return [self.snr_conditions[c] for c in snr_cond]

    def frame_to_token_emb(
        self,
        frame_emb: torch.Tensor,
        frame_len: torch.Tensor,
        token_duration: torch.Tensor,
        token_len: torch.Tensor,
    ) -> torch.Tensor:
        """Average frame-level SNR embeddings inside the span of each token (short-term feedback).

        Args:
            frame_emb: (batch, frame_maxlen, emb_dim) at the frontend rate of the SNR predictor
            frame_len: (batch,)
            token_duration: (batch, token_maxlen) durations in frames of the TTS decoder
            token_len: (batch,)

        Returns:
            (batch, token_maxlen, emb_dim)
        """
        batch_size, token_maxlen = token_duration.size()
        token_emb = torch.zeros(
            batch_size, token_maxlen, frame_emb.size(-1), device=frame_emb.device
        )
        for i in range(batch_size):
            dur = token_duration[i, : token_len[i]].float().clamp(min=0)
            total = dur.sum().clamp(min=1)
            # map token boundaries onto the frame axis of the SNR predictor
            bounds = (torch.cumsum(dur, dim=0) / total * frame_len[i]).round().long()
            starts = torch.cat([bounds.new_zeros(1), bounds[:-1]])
            n_frames = int(frame_len[i])
            if n_frames <= 0:
                continue
            for j in range(int(token_len[i])):
                # clamp the span to the valid frames so that every token averages
                # at least one frame (zero-duration or trailing tokens would otherwise
                # give an empty slice and NaN)
                s = min(int(starts[j]), n_frames - 1)
                e = min(max(int(bounds[j]), s + 1), n_frames)
                token_emb[i, j] = frame_emb[i, s:e].mean(dim=0)
        return token_emb

    # --- Model forward --- #
    def module_forward(
        self,
        epoch: int = None,
        text: torch.Tensor = None,
        text_len: torch.Tensor = None,
        feat: torch.Tensor = None,
        feat_len: torch.Tensor = None,
        feat_ref: torch.Tensor = None,
        feat_ref_len: torch.Tensor = None,
        text_asr: List[str] = None,
        snr_cond: List[str] = None,
        feedback: Dict = None,
        snr_coeff: float = None,
        asr_coeff: float = None,
        **kwargs,
    ) -> Dict:
        """
        Args:
            feat_ref, feat_ref_len:
                The normal (non-Lombard) waveforms of the sentences used to obtain the feedback
                during training. If not given, `feat` is used.
            text_asr: List[str]
                The raw transcripts for the ASR loss.
            snr_cond: List[str]
                The noise condition names of the utterances.
            feedback: Dict
                Pre-computed feedback (given by inference()). If None during training/validation,
                the feedback is obtained by listening to the reference speech in the noise of
                `snr_cond`.
            snr_coeff, asr_coeff: float
                Override the coefficients of the feedback embeddings.
            The other arguments follow FastSpeech2.module_forward().
        """
        # feedback stored by inference() for the calls made through FastSpeech2.inference()
        pending = getattr(self, "_pending_feedback", None)
        if feedback is None and pending is not None:
            feedback, snr_coeff, asr_coeff = pending

        # --- 1. Auditory feedback --- #
        snr_tgt = None
        if feedback is None and feat is not None and snr_cond is not None:
            ref, ref_len = (
                (feat, feat_len) if feat_ref is None else (feat_ref, feat_ref_len)
            )
            noise_type = [
                self.train_noise_types[
                    int(torch.randint(len(self.train_noise_types), (1,)))
                ]
                for _ in range(len(snr_cond))
            ]
            feedback = self.listen(
                ref,
                ref_len,
                self.sample_rate,
                self.snr_spec_of_cond(snr_cond),
                noise_type,
                text_asr,
            )
            snr_tgt = self.snr_predictor.class_ids(snr_cond).to(text.device)

        # --- 2. Encoder + feedback embedding --- #
        # remove the <sos/eos> at the beginning and the end of each sentence (as in FastSpeech2)
        for i in range(text_len.size(0)):
            text[i, text_len[i] - 1] = self.tokenizer.ignore_idx
        text, text_len = text[:, 1:-1], text_len - 2
        enc_text, enc_text_mask, enc_attmat, enc_hidden = self.encoder(
            text=text, text_len=text_len
        )

        if feedback is not None:
            snr_emb = feedback.get("snr_token_emb", feedback.get("snr_emb", None))
            if snr_emb is not None and self.snr_emb_detach:
                snr_emb = snr_emb.detach()
            asr_loss = feedback.get("asr_loss", None)
            enc_text = self.feedback_embed(
                enc_text,
                snr_emb=snr_emb,
                asr_loss=asr_loss.detach() if asr_loss is not None else None,
                snr_coeff=snr_coeff,
                asr_coeff=asr_coeff,
            )

        # --- 3. Decoder (variance adaptor + mel decoder) --- #
        dec_args = {
            k: kwargs.get(k, None)
            for k in [
                "duration",
                "duration_len",
                "pitch",
                "pitch_len",
                "energy",
                "energy_len",
                "spk_feat",
                "spk_ids",
                "duration_alpha",
                "energy_alpha",
                "pitch_alpha",
            ]
        }
        (
            pred_feat_before,
            pred_feat_after,
            pred_feat_len,
            tgt_feat,
            tgt_feat_len,
            pred_pitch,
            tgt_pitch,
            tgt_pitch_len,
            pred_energy,
            tgt_energy,
            tgt_energy_len,
            pred_duration,
            pred_duration_gate,
            tgt_duration,
            tgt_duration_len,
            dec_attmat,
            dec_hidden,
        ) = self.decoder(
            enc_text=enc_text,
            enc_text_mask=enc_text_mask,
            feat=feat,
            feat_len=feat_len,
            epoch=epoch,
            min_frame_num=kwargs.get("min_frame_num", 0),
            max_frame_num=kwargs.get("max_frame_num", None),
            **dec_args,
        )

        outputs = dict(
            pred_feat_before=pred_feat_before,
            pred_feat_after=pred_feat_after,
            pred_feat_len=pred_feat_len,
            tgt_feat=tgt_feat,
            tgt_feat_len=tgt_feat_len,
            pred_pitch=pred_pitch,
            tgt_pitch=tgt_pitch,
            tgt_pitch_len=tgt_pitch_len,
            pred_energy=pred_energy,
            tgt_energy=tgt_energy,
            tgt_energy_len=tgt_energy_len,
            pred_duration=pred_duration,
            pred_duration_gate=pred_duration_gate,
            tgt_duration=tgt_duration,
            tgt_duration_len=tgt_duration_len,
        )
        if feedback is not None:
            outputs.update(
                snr_logits=feedback.get("snr_logits", None),
                fb_asr_loss=feedback.get("asr_loss", None),
                fb_snr_applied=feedback.get("snr_applied", None),
            )
        if snr_tgt is not None:
            outputs.update(snr_tgt=snr_tgt)

        if kwargs.get("return_att", False):
            att = dict()
            if enc_attmat is not None and "enc" in self.return_att_type:
                att["enc"] = enc_attmat[-self.return_att_layer_num :]
            if dec_attmat is not None and "dec" in self.return_att_type:
                att["dec"] = dec_attmat[-self.return_att_layer_num :]
            outputs.update(att=att)
        return outputs

    # --- Criteria --- #
    def criterion_init(self, snr_loss_weight: float = 1.0, **fs2_criterion_conf):
        super().criterion_init(**fs2_criterion_conf)
        self.snr_loss_weight = snr_loss_weight
        self.snr_loss = torch.nn.CrossEntropyLoss()

    def criterion_forward(
        self,
        snr_logits: torch.Tensor = None,
        snr_tgt: torch.Tensor = None,
        fb_asr_loss: torch.Tensor = None,
        fb_snr_applied: torch.Tensor = None,
        **kwargs,
    ):
        results = super().criterion_forward(**kwargs)
        losses, metrics = results if self.training else (None, results)

        if snr_logits is not None and snr_tgt is not None:
            snr_loss = self.snr_loss(snr_logits.float(), snr_tgt)
            snr_acc = (snr_logits.argmax(dim=-1) == snr_tgt).float().mean()
            metrics.update(snr_loss=snr_loss.clone().detach(), snr_acc=snr_acc.detach())
            if self.training:
                losses["loss"] = losses["loss"] + self.snr_loss_weight * snr_loss
                metrics["loss"] = losses["loss"].clone().detach()
        if fb_asr_loss is not None:
            metrics.update(fb_asr_loss=fb_asr_loss.mean().detach())

        return (losses, metrics) if self.training else metrics

    # --- Visualization during training --- #
    def visualize(self, epoch: int, sample_index: str, **kwargs):
        for key in ["feat_ref", "feat_ref_len", "text_asr", "snr_cond"]:
            kwargs.pop(key, None)
        # without text_asr/snr_cond, there is no feedback signal to adapt to, so force the cheap
        # single-pass standard-TTS path (max_loops=0 AND eval_asr=False -- inference() only takes
        # that shortcut when both hold); otherwise the default visual_infer_conf (which sets
        # neither) falls through to the full closed loop, which is both wasteful and drops 'att'
        # from the returned dict that FastSpeech2.visualize() needs
        if len(self.visual_infer_conf) == 0:
            self.visual_infer_conf = dict(
                teacher_forcing=False,
                return_wav=False,
                return_feat=True,
                max_loops=0,
                eval_asr=False,
            )
        return super().visualize(epoch=epoch, sample_index=sample_index, **kwargs)

    # --- Dynamically adaptive inference --- #
    def inference(
        self,
        infer_conf: Dict,
        text: torch.Tensor = None,
        text_len: torch.Tensor = None,
        text_asr: List[str] = None,
        snr_cond: List[str] = None,
        feat: torch.Tensor = None,
        feat_len: torch.Tensor = None,
        pitch: torch.Tensor = None,
        pitch_len: torch.Tensor = None,
        duration: torch.Tensor = None,
        duration_len: torch.Tensor = None,
        spk_ids: torch.Tensor = None,
        spk_feat: torch.Tensor = None,
        spk_feat_ids: List[str] = None,
        domain: str = None,
        return_att: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, str or List]]:
        """Closed-loop machine speech chain inference.

        Lombard-specific keys of `infer_conf` (the others are passed to FastSpeech2.inference()):
            snr: SNRSpec or str
                The noise environment: a SNR in dB, a dynamic profile [[start_ratio, snr_db], ...],
                the name of a training condition, or null for the clean condition.
                Default: the `snr_cond` given by the dataset, or clean.
            noise_type: str
                'white' or a registered noise file name. Default: 'white'.
            max_loops: int
                Maximum number of feedback loops. 0 means standard TTS without feedback.
            loss_tol: float
                The loop stops when the ASR loss decreases by less than loss_tol.
            select_best: bool
                Return the output of the loop with the lowest ASR loss (otherwise the last one).
            feedback_level: 'utterance' or 'token'
                Utterance-level or token-level (short-term) SNR feedback.
            snr_coeff, asr_coeff: float
                Coefficients of the feedback embeddings.
            eval_asr: bool
                Decode the final noisy speech with the ASR and report CER/WER against text_asr.
            asr_decode_conf: Dict
                The decoding configuration of the ASR (default: greedy search).
            return_noisy_wav: bool
                Also return the noisy version of the final speech.
        """
        assert text is not None and text_len is not None
        infer_conf = copy.deepcopy(infer_conf)
        snr = infer_conf.pop("snr", "__from_batch__")
        noise_type = infer_conf.pop("noise_type", "white")
        max_loops = infer_conf.pop("max_loops", 4)
        loss_tol = infer_conf.pop("loss_tol", 0.01)
        select_best = infer_conf.pop("select_best", True)
        feedback_level = infer_conf.pop("feedback_level", "utterance")
        snr_coeff = infer_conf.pop("snr_coeff", None)
        asr_coeff = infer_conf.pop("asr_coeff", None)
        eval_asr = infer_conf.pop("eval_asr", True)
        asr_decode_conf = infer_conf.pop("asr_decode_conf", dict(beam_size=1))
        return_noisy_wav = infer_conf.pop("return_noisy_wav", True)
        assert feedback_level in ["utterance", "token"]
        # the loop needs waveforms; teacher-forcing is left to the parent class as-is
        teacher_forcing = infer_conf.get("teacher_forcing", False)
        if teacher_forcing or max_loops == 0 and not eval_asr:
            return super().inference(
                infer_conf,
                text=text,
                text_len=text_len,
                feat=feat,
                feat_len=feat_len,
                pitch=pitch,
                pitch_len=pitch_len,
                duration=duration,
                duration_len=duration_len,
                spk_ids=spk_ids,
                spk_feat=spk_feat,
                spk_feat_ids=spk_feat_ids,
                domain=domain,
                return_att=return_att,
            )
        infer_conf["return_wav"] = True
        return_feat = infer_conf.get("return_feat", False)
        # applied to the final selected wav below, not threaded through every loop's synthesis
        # (the closed loop always needs native-rate audio internally for the feedback listener)
        return_sr = infer_conf.pop("return_sr", None)

        batch_size = text.size(0)
        if snr == "__from_batch__":
            snr_list = (
                self.snr_spec_of_cond(snr_cond)
                if snr_cond is not None
                else [None] * batch_size
            )
        elif isinstance(snr, str):
            snr_list = [self.snr_conditions[snr]] * batch_size
        else:
            snr_list = [snr] * batch_size
        if text_asr is None:
            eval_asr = False

        def synthesize(feedback):
            self._pending_feedback = (feedback, snr_coeff, asr_coeff)
            try:
                return super(LombardFastSpeech2, self).inference(
                    infer_conf,
                    text=text.clone(),
                    text_len=text_len.clone(),
                    spk_ids=spk_ids,
                    spk_feat=spk_feat,
                    spk_feat_ids=spk_feat_ids,
                    domain=domain,
                    return_att=return_att,
                )
            finally:
                self._pending_feedback = None

        def wav_batch(outputs):
            wavs = [
                torch.as_tensor(w, device=text.device).squeeze(-1)
                for w in outputs["wav"]["content"]
            ]
            wav_len = torch.LongTensor([w.size(0) for w in wavs]).to(text.device)
            wav = torch.zeros(batch_size, int(wav_len.max()), device=text.device)
            for i, w in enumerate(wavs):
                wav[i, : w.size(0)] = w
            return wav, wav_len

        # loop 0: speak without any feedback
        history = []
        outputs = synthesize(None)
        wav, wav_len = wav_batch(outputs)
        feedback = self.listen(
            wav, wav_len, self.sample_rate, snr_list, noise_type, text_asr
        )
        history.append(dict(outputs=outputs, feedback=feedback, loop=0))

        for loop in range(1, max_loops + 1):
            if feedback_level == "token":
                feedback["snr_token_emb"] = self.frame_to_token_emb(
                    feedback["snr_frame_emb"],
                    feedback["snr_frame_len"],
                    torch.nn.utils.rnn.pad_sequence(
                        [torch.as_tensor(d) for d in outputs["duration"]["content"]],
                        batch_first=True,
                    ).to(text.device),
                    text_len - 2,
                )
            outputs = synthesize(feedback)
            wav, wav_len = wav_batch(outputs)
            new_feedback = self.listen(
                wav, wav_len, self.sample_rate, snr_list, noise_type, text_asr
            )
            history.append(dict(outputs=outputs, feedback=new_feedback, loop=loop))
            if (
                new_feedback["asr_loss"] is not None
                and feedback["asr_loss"] is not None
            ):
                improvement = (
                    (feedback["asr_loss"] - new_feedback["asr_loss"]).mean().item()
                )
                feedback = new_feedback
                if improvement < loss_tol:
                    break
            else:
                feedback = new_feedback

        # --- select the final output of each utterance --- #
        loss_matrix = (
            torch.stack([h["feedback"]["asr_loss"] for h in history])
            if history[0]["feedback"]["asr_loss"] is not None
            else None
        )
        if select_best and loss_matrix is not None:
            best_loop = loss_matrix.argmin(dim=0)
        else:
            best_loop = torch.full(
                (batch_size,), len(history) - 1, dtype=torch.long, device=text.device
            )

        final = dict()
        for key in [
            "wav",
            "wav_len",
            "feat",
            "feat_len",
            "duration",
            "feat_token_len_ratio",
        ]:
            if key in history[0]["outputs"]:
                final[key] = copy.copy(history[0]["outputs"][key])
                final[key]["content"] = [
                    history[int(best_loop[i])]["outputs"][key]["content"][i]
                    for i in range(batch_size)
                ]
        if return_sr is not None and return_sr != self.sample_rate and "wav" in final:
            if not hasattr(self, "output_resampler_cache"):
                self.output_resampler_cache = {}
            resampler = get_cached_resampler(
                self.output_resampler_cache,
                self.sample_rate,
                return_sr,
                device=text.device,
            )
            resampled_wav = []
            for w in final["wav"]["content"]:
                wav_t = torch.as_tensor(w, device=text.device).float()
                wav_t = wav_t.squeeze(-1) if wav_t.dim() == 2 else wav_t
                resampled_wav.append(resampler(wav_t))
            final["wav"]["content"] = to_cpu(resampled_wav, tgt="numpy")
            final["wav"]["sample_rate"] = return_sr
            if "wav_len" in final:
                final["wav_len"]["content"] = to_cpu(
                    torch.LongTensor([w.size(0) for w in resampled_wav])
                )
        if not return_feat:
            final.pop("feat", None), final.pop("feat_len", None)

        noisy_list, noisy_len = [], []
        for i in range(batch_size):
            fb = history[int(best_loop[i])]["feedback"]
            noisy_list.append(fb["noisy_wav"][i, : fb["noisy_wav_len"][i]])
            noisy_len.append(int(fb["noisy_wav_len"][i]))
        if return_noisy_wav:
            final["noisy_wav"] = dict(
                format="wav",
                sample_rate=self.feedback_sample_rate,
                content=to_cpu([w.unsqueeze(-1) for w in noisy_list], tgt="numpy"),
            )

        final["loops_used"] = dict(format="txt", content=to_cpu(best_loop))
        final["snr_pred"] = dict(
            format="txt",
            content=[
                self.snr_classes[
                    int(
                        history[int(best_loop[i])]["feedback"]["snr_logits"][i].argmax()
                    )
                ]
                for i in range(batch_size)
            ],
        )
        final["snr_applied"] = dict(
            format="txt",
            content=[
                f"{float(history[int(best_loop[i])]['feedback']['snr_applied'][i]):.2f}"
                for i in range(batch_size)
            ],
        )
        if loss_matrix is not None:
            final["asr_loss_per_loop"] = dict(
                format="txt",
                content=[
                    str([round(float(v), 4) for v in loss_matrix[:, i]])
                    for i in range(batch_size)
                ],
            )
            final["asr_loss"] = dict(
                format="txt",
                content=to_cpu(
                    loss_matrix[
                        best_loop,
                        torch.arange(batch_size, device=loss_matrix.device),
                    ]
                ),
            )
            final["asr_loss_loop0"] = dict(format="txt", content=to_cpu(loss_matrix[0]))

        instance_report = {"Loop": [str(int(l)) for l in best_loop]}
        # --- evaluate the intelligibility of the final (and the loop-0 baseline) noisy speech by the ASR --- #
        if eval_asr:
            asr = self.asr
            for tag, noisy_wavs, lens in [
                ("", noisy_list, noisy_len),
                (
                    "_loop0",
                    [
                        history[0]["feedback"]["noisy_wav"][
                            i, : history[0]["feedback"]["noisy_wav_len"][i]
                        ]
                        for i in range(batch_size)
                    ],
                    [int(v) for v in history[0]["feedback"]["noisy_wav_len"]],
                ),
            ]:
                padded = torch.zeros(batch_size, max(lens), device=text.device)
                for i, w in enumerate(noisy_wavs):
                    padded[i, : w.size(0)] = w
                with torch.inference_mode():
                    asr_out = asr.inference(
                        copy.deepcopy(asr_decode_conf),
                        feat=padded.unsqueeze(-1),
                        feat_len=torch.LongTensor(lens).to(text.device),
                        decode_only=True,
                    )
                hypo = asr_out["text"]["content"]
                cer, wer = asr.error_rate(
                    hypo_text=list(hypo), real_text=list(text_asr)
                )
                final[f"hypo_text{tag}"] = dict(format="txt", content=hypo)
                final[f"cer{tag}"] = dict(format="txt", content=[float(c) for c in cer])
                final[f"wer{tag}"] = dict(format="txt", content=[float(w) for w in wer])
                instance_report[f"CER{tag}"] = [f"{float(c):.4f}" for c in cer]
                if tag == "":
                    instance_report["Hypothesis"] = hypo
            instance_report["Reference"] = list(text_asr)
        self.register_instance_reports(md_list_dict=instance_report)
        return final
