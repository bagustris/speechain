import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("pyworld")

from speechain.model.lombard_tts import LombardFastSpeech2

PHONES = ["<sos/eos>", "<blank>", "<unk>", "<space>", "AH0", "K", "T", "DH"]


class StubTokenizer:
    """Character-level tokenizer stub for the ASR listener."""

    def __init__(self):
        self.vocab = ["<blank>", "<sos/eos>", " ", "a", "b", "c", "t", "h", "e"]
        self.ignore_idx, self.sos_eos_idx = 0, 1

    def text2tensor(self, text, no_sos=False, no_eos=False, return_tensor=True):
        ids = [self.vocab.index(c) if c in self.vocab else 2 for c in text]
        return torch.LongTensor([self.sos_eos_idx] + ids + [self.sos_eos_idx])


class StubASR(torch.nn.Module):
    """Minimal ASR listener: random logits, fixed hypotheses, CER from the ErrorRate criterion."""

    def __init__(self):
        super().__init__()
        self.tokenizer = StubTokenizer()
        self.proj = torch.nn.Linear(1, len(self.tokenizer.vocab))
        self.device = torch.device("cpu")
        self.calls = []

    def module_forward(self, feat, feat_len, text, text_len, **kwargs):
        self.calls.append(feat_len.clone())
        return dict(logits=self.proj(torch.zeros(text.size(0), text.size(1) - 1, 1)))

    def inference(self, infer_conf, feat, feat_len, decode_only=True, **kwargs):
        return dict(
            text=dict(format="txt", content=["the cat" for _ in range(feat.size(0))])
        )

    def error_rate(self, hypo_text, real_text):
        from speechain.criterion.error_rate import ErrorRate

        # mirror ar_asr.py, which binds the tokenizer when it sets self.error_rate = ErrorRate(tokenizer=self.tokenizer)
        return ErrorRate()(
            hypo_text=hypo_text, real_text=real_text, tokenizer=self.tokenizer
        )


@pytest.fixture(scope="module")
def token_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("tokens")
    (path / "vocab").write_text("\n".join(PHONES) + "\n")
    return str(path)


def build_model(token_path, tmp_path, **customize):
    d = 16
    frontend = dict(
        type="frontend.speech2mel.Speech2MelSpec",
        conf=dict(
            sr=16000,
            mag_spec=True,
            hop_length=256,
            win_length=1024,
            n_mels=20,
            fmin=0,
            fmax=8000,
            log_base=None,
            clamp=1e-5,
        ),
    )
    var_pred = dict(
        type="prenet.var_pred.Conv1dVarPredictor",
        conf=dict(conv_dims=[8, -1], conv_kernel=3, conv_emb_kernel=1),
    )
    trm = dict(
        type="transformer.encoder.TransformerEncoder",
        conf=dict(d_model=d, num_heads=2, num_layers=1, fdfwd_dim=32),
    )
    module_conf = dict(
        enc_emb=dict(type="prenet.embed.EmbedPrenet", conf=dict(embedding_dim=d)),
        encoder=trm,
        duration_predictor=dict(
            type="prenet.var_pred.Conv1dVarPredictor",
            conf=dict(conv_dims=[8, -1], conv_kernel=3),
        ),
        pitch_predictor=var_pred,
        energy_predictor=var_pred,
        feat_frontend=frontend,
        decoder=trm,
        dec_postnet=dict(
            type="postnet.conv1d.Conv1dPostnet",
            conf=dict(conv_dims=[8, 0], conv_kernel=3),
        ),
        snr_predictor=dict(
            frontend=dict(
                type="frontend.speech2mel.Speech2MelSpec",
                conf=dict(sr=16000, hop_length=0.01, win_length=0.025, n_mels=20),
            ),
            conv_dims=[8],
        ),
    )
    customize_conf = dict(
        token_type="mfa",
        token_path=token_path,
        sample_rate=16000,
        snr_conditions={"clean": None, "snr0": 0, "snr-10": -10},
        asr_model=StubASR(),
    )
    customize_conf.update(customize)
    model = LombardFastSpeech2(
        device=torch.device("cpu"),
        module_conf=module_conf,
        model_conf=dict(customize_conf=customize_conf),
        criterion_conf=dict(snr_loss_weight=0.5),
        result_path=str(tmp_path),
    )
    # the base class always moves batches to CUDA; keep them on the CPU for the tests
    model.batch_to_cuda = lambda batch: batch
    # the duration predictor is otherwise randomly initialized around a 0 log-duration bias
    # (exp(0) - 1 rounds to 0 frames/token), which produces synthetic speech too short for the
    # GL vocoder (needs >= win_length samples); bias it to a safe non-degenerate duration so
    # inference tests exercise the actual closed-loop/vocoding logic instead of a GL failure
    torch.nn.init.constant_(model.decoder.duration_predictor.linear.bias, 1.6)
    return model


def fake_batch():
    torch.manual_seed(0)
    phon = ["DH", "AH0", "<space>", "K", "AH0", "T"]
    return dict(
        feat=torch.randn(2, 16000, 1) * 0.1,
        feat_len=torch.tensor([16000, 12000]),
        feat_ref=torch.randn(2, 16000, 1) * 0.1,
        feat_ref_len=torch.tensor([16000, 12000]),
        text=[phon, phon[:4]],
        duration=torch.tensor(
            [[0.1, 0.2, 0.05, 0.2, 0.3, 0.15], [0.2, 0.3, 0.1, 0.4, 0, 0]]
        ),
        duration_len=torch.tensor([6, 4]),
        pitch=torch.rand(2, 63) * 200,
        pitch_len=torch.tensor([63, 47]),
        text_asr=["the cat", "the"],
        snr_cond=["snr0", "clean"],
    )


class TestLombardFastSpeech2:
    def test_training_step(self, token_path, tmp_path):
        model = build_model(token_path, tmp_path)
        model.train()
        losses, metrics = model(fake_batch())
        for key in [
            "feat_loss_after",
            "pitch_loss",
            "energy_loss",
            "duration_loss",
            "snr_loss",
            "snr_acc",
            "fb_asr_loss",
        ]:
            assert key in metrics, key
        assert losses["loss"].requires_grad
        losses["loss"].backward()
        # the SNR predictor is trained by its classification loss
        assert model.snr_predictor.classifier.weight.grad is not None
        # the frozen listener is not part of the checkpoint
        assert not any(k.startswith("_asr") for k in model.state_dict().keys())
        assert all(not p.requires_grad for p in model.asr.parameters())

    def test_validation_step(self, token_path, tmp_path):
        model = build_model(token_path, tmp_path)
        # global feature normalization stats are only accumulated during training-mode
        # forward passes, so warm them up before switching to eval (mirrors real usage,
        # where inference always follows training or loading a trained checkpoint)
        model.train()
        model(fake_batch())
        model.eval()
        metrics = model(fake_batch())
        assert "snr_acc" in metrics and not metrics["loss"].requires_grad

    def test_closed_loop_inference(self, token_path, tmp_path):
        model = build_model(token_path, tmp_path)
        model.train()
        model(fake_batch())
        model.eval()
        phon = ["DH", "AH0", "<space>", "K", "AH0", "T"]
        batch = dict(
            text=[phon, phon[:4]],
            text_asr=["the cat", "the"],
            snr_cond=["snr0", "snr-10"],
        )
        out = model.evaluate(
            batch, infer_conf=dict(vocoder="gl", max_loops=2, loss_tol=-1.0)
        )
        for key in [
            "wav",
            "noisy_wav",
            "loops_used",
            "snr_pred",
            "asr_loss_per_loop",
            "cer",
            "cer_loop0",
            "hypo_text",
        ]:
            assert key in out, key
        assert out["noisy_wav"]["sample_rate"] == 16000
        # loop 0 + 2 feedback loops were listened to
        assert all(len(eval(v)) == 3 for v in out["asr_loss_per_loop"]["content"])
        assert (
            out["cer"]["content"][0] == 0.0
        )  # the stub hypothesis 'the cat' matches the first reference
        assert out["cer"]["content"][1] > 0.0

    def test_dynamic_noise_token_level(self, token_path, tmp_path):
        model = build_model(token_path, tmp_path)
        model.train()
        model(fake_batch())
        model.eval()
        phon = ["DH", "AH0", "K", "AH0", "T"]
        out = model.evaluate(
            dict(text=[phon]),
            infer_conf=dict(
                vocoder="gl",
                max_loops=1,
                snr=[[0.0, 10.0], [0.5, -10.0]],
                feedback_level="token",
                eval_asr=False,
            ),
        )
        assert "cer" not in out and out["snr_applied"]["content"][0] != "inf"

    def test_standard_tts_without_feedback(self, token_path, tmp_path):
        model = build_model(token_path, tmp_path)
        model.train()
        model(fake_batch())
        model.eval()
        out = model.evaluate(
            dict(text=[["K", "AH0", "T"]]),
            infer_conf=dict(vocoder="gl", max_loops=0, eval_asr=False),
        )
        assert "wav" in out and "asr_loss" not in out

    def test_return_sr_in_closed_loop(self, token_path, tmp_path):
        model = build_model(token_path, tmp_path)
        model.train()
        model(fake_batch())
        model.eval()
        phon = ["DH", "AH0", "<space>", "K", "AH0", "T"]
        batch = dict(
            text=[phon, phon[:4]],
            text_asr=["the cat", "the"],
            snr_cond=["snr0", "snr-10"],
        )
        out = model.evaluate(
            batch,
            infer_conf=dict(vocoder="gl", max_loops=2, loss_tol=-1.0, return_sr=8000),
        )
        # the closed loop must honor a requested output rate on the final selected wav,
        # not silently return audio at the model's native rate (see PR review comment)
        assert out["wav"]["sample_rate"] == 8000
        assert all(len(w) > 0 for w in out["wav"]["content"])

    def test_frame_to_token_emb(self, token_path, tmp_path):
        model = build_model(token_path, tmp_path)
        frame_emb = torch.arange(10.0).view(1, 10, 1)
        token_emb = model.frame_to_token_emb(
            frame_emb, torch.tensor([10]), torch.tensor([[5, 5]]), torch.tensor([2])
        )
        assert token_emb.squeeze().tolist() == [2.0, 7.0]

    @pytest.mark.parametrize(
        "durations", [[5, 0, 0], [0, 0, 0], [0, 5, 0], [10, 0, 0, 0]]
    )
    def test_frame_to_token_emb_zero_duration_spans(
        self, token_path, tmp_path, durations
    ):
        # zero-duration / trailing tokens must still average at least one frame
        model = build_model(token_path, tmp_path)
        frame_emb = torch.arange(10.0).view(1, 10, 1)
        n_tokens = len(durations)
        token_emb = model.frame_to_token_emb(
            frame_emb,
            torch.tensor([10]),
            torch.tensor([durations]),
            torch.tensor([n_tokens]),
        )
        assert token_emb.shape == (1, n_tokens, 1)
        assert torch.isfinite(token_emb).all()
