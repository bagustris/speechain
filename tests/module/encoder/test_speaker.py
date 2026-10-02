import pytest

torch = pytest.importorskip("torch")

from speechain.module.encoder.speaker import EncoderClassifier, Res2NetBlock, SEBlock


class TestEncoderClassifier:
    def setup_method(self):
        self.device = torch.device("cpu")
        self.ecapa_model = EncoderClassifier(model_type="ecapa")
        self.xvector_model = EncoderClassifier(model_type="xvector")

    def test_seblock_forward(self):
        se = SEBlock(in_channels=512, se_channels=128, out_channels=512)
        x = torch.randn(2, 512, 100)
        out = se(x)
        assert out.shape == x.shape

    def test_res2netblock_forward(self):
        res = Res2NetBlock(in_channels=512, out_channels=512)
        x = torch.randn(2, 512, 100)
        out = res(x)
        assert out.shape == x.shape

    def test_ecapa_forward(self):
        batch_size = 2
        seq_len = 100
        mel_channels = 80
        # embedding_model expects (batch, time, channel)
        x = torch.randn(batch_size, seq_len, mel_channels)
        out = self.ecapa_model.embedding_model(x)
        assert out.shape == (batch_size, 1, 192)

    def test_xvector_forward(self):
        batch_size = 2
        seq_len = 100
        mel_channels = 24
        # embedding_model expects (batch, time, channel)
        x = torch.randn(batch_size, seq_len, mel_channels)
        out = self.xvector_model.embedding_model(x)
        assert out.shape == (batch_size, 1, 512)

    def test_encode_batch(self):
        batch_size = 2
        seq_len = 16000  # 1 second of raw waveform at 16kHz (encode_batch expects raw waveforms)
        x = torch.randn(batch_size, seq_len)
        out = self.ecapa_model.encode_batch(x)
        assert out.shape == (batch_size, 1, 192)
        assert torch.allclose(
            torch.norm(out.squeeze(1), p=2, dim=1), torch.ones(batch_size), atol=1e-5
        )

    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            EncoderClassifier(model_type="invalid")

    def test_from_hparams_ecapa(self):
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": self.device}
        )
        assert model.model_type == "ecapa"
        assert next(model.parameters()).device == self.device

    def test_from_hparams_keeps_norm_stats_dimension(self, tmp_path, monkeypatch):
        # offline: serve a local checkpoint and normalization stats instead of the hub
        emb_dim = 192
        ref = EncoderClassifier(model_type="ecapa")
        ckpt = tmp_path / "embedding_model.ckpt"
        torch.save(ref.embedding_model.state_dict(), ckpt)
        norm = tmp_path / "mean_var_norm_emb.ckpt"
        torch.save(
            {"glob_mean": torch.randn(emb_dim), "glob_std": torch.rand(emb_dim) + 0.5},
            norm,
        )

        def fake_download(repo_id, filename, cache_dir=None, **kwargs):
            return str(ckpt if filename == "embedding_model.ckpt" else norm)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": self.device}
        )
        assert model.glob_mean.shape == (1, 1, emb_dim)
        assert model.glob_std.shape == (1, 1, emb_dim)
        emb = model.encode_batch(torch.randn(1, 16000), normalize=True)
        assert emb.shape == (1, 1, emb_dim)

    def test_from_hparams_xvector(self):
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": self.device}
        )
        assert model.model_type == "xvector"
        assert next(model.parameters()).device == self.device
