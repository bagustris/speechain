# Vocoder and Speaker Embedding in SpeeChain

Starting from v0.2, SpeeChain has removed the SpeechBrain dependency and implemented its own versions of the following components:

## HiFiGAN Vocoder

SpeeChain includes a custom implementation of the HiFiGAN vocoder in the `speechain.module.vocoder` module. This vocoder is used to convert mel-spectrograms into waveforms for text-to-speech synthesis.

The vocoder can be used with both single-speaker and multi-speaker settings:

```python
from speechain.utilbox.vocoder_util import get_hifigan_vocoder

# For single-speaker TTS (LJSpeech)
vocoder = get_hifigan_vocoder(
    device="cuda:0",
    sample_rate=22050,
    use_multi_speaker=False
)

# For multi-speaker TTS (LibriTTS)
vocoder = get_hifigan_vocoder(
    device="cuda:0",
    sample_rate=22050,
    use_multi_speaker=True
)

# To generate waveform from mel-spectrogram
waveform, waveform_len = vocoder(mel_spectrogram, mel_spectrogram_length)
```

## Speaker Encoder

For multi-speaker TTS, SpeeChain includes a custom implementation of both ECAPA-TDNN and X-vector speaker embedding models in the `speechain.module.encoder.speaker` module.

These embeddings are used to control the speaker identity in multi-speaker TTS systems:

```python
from speechain.module.encoder.speaker import EncoderClassifier

# Create an ECAPA-TDNN model
speaker_encoder = EncoderClassifier(model_type="ecapa")

# Or create an X-vector model
speaker_encoder = EncoderClassifier(model_type="xvector")

# Extract speaker embeddings from waveforms
embeddings = speaker_encoder.encode_batch(wavs, wav_lens)
```

## Speaker Embedding Extraction Utility

SpeeChain provides a utility for extracting speaker embeddings from audio files:

```python
from speechain.utilbox.spk_util import extract_spk_feat

# Extract speaker embeddings for a collection of speaker audio files
idx2spk_feat, spk2aver_spk_feat = extract_spk_feat(
    spk2wav_dict=speaker_audio_dict,
    gpu_id=0,
    spk_emb_model="ecapa",
    save_path="path/to/save/embeddings"
)
```

## Model Weights

The pre-trained weights for these models are automatically downloaded when the models are first used. The weights are compatible with SpeechBrain's pre-trained models but are used with SpeeChain's own implementation.
