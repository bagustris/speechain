#!/usr/bin/env python3
"""
Test script for SpeeChain's own vocoder and speaker embedding implementations.
This script verifies that these components work correctly without relying on SpeechBrain.
"""

import argparse
import os
import torch
import torchaudio

from speechain.module.encoder.speaker import EncoderClassifier
from speechain.module.vocoder import HIFIGAN
from speechain.utilbox.vocoder_util import get_hifigan_vocoder, VocoderWrapper


def test_vocoder(wav_path, output_dir):
    """Test the HiFiGAN vocoder by converting a waveform to mel-spec and back."""
    print("Testing HiFiGAN vocoder...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load an audio file
    wav, sr = torchaudio.load(wav_path)
    if sr != 22050:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
        wav = resampler(wav)
        sr = 22050
    
    # Generate a mel-spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )
    mel = mel_transform(wav)
    
    # Get the vocoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocoder = get_hifigan_vocoder(device=device, sample_rate=sr, use_multi_speaker=True)
    
    # Convert mel-spectrogram back to waveform
    wav_len = torch.tensor([mel.shape[2]], device=device)
    reconstructed_wav, reconstructed_len = vocoder(mel, wav_len)
    
    # Save the reconstructed waveform
    output_path = os.path.join(output_dir, "reconstructed.wav")
    torchaudio.save(output_path, reconstructed_wav[0].cpu(), sr)
    
    print(f"Original waveform shape: {wav.shape}")
    print(f"Mel-spectrogram shape: {mel.shape}")
    print(f"Reconstructed waveform shape: {reconstructed_wav.shape}")
    print(f"Reconstructed waveform saved to: {output_path}")
    
    return output_path


def test_speaker_encoder(wav_path):
    """Test the speaker encoder by extracting speaker embeddings from a waveform."""
    print("\nTesting Speaker Encoder...")
    
    # Load an audio file
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)
        sr = 16000
    
    # Initialize both types of speaker encoders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ecapa_encoder = EncoderClassifier(model_type="ecapa")
    xvector_encoder = EncoderClassifier(model_type="xvector")
    
    ecapa_encoder = ecapa_encoder.to(device)
    xvector_encoder = xvector_encoder.to(device)
    
    # Extract speaker embeddings
    wav = wav.to(device)
    wav_len = torch.tensor([wav.shape[1]], device=device).float() / wav.shape[1]
    
    with torch.no_grad():
        ecapa_embedding = ecapa_encoder.encode_batch(wav, wav_len)
        xvector_embedding = xvector_encoder.encode_batch(wav, wav_len)
    
    print(f"ECAPA-TDNN embedding shape: {ecapa_embedding.shape}")
    print(f"X-vector embedding shape: {xvector_embedding.shape}")
    
    return ecapa_embedding, xvector_embedding


def main():
    parser = argparse.ArgumentParser(description="Test SpeeChain vocoder and speaker embedding modules")
    parser.add_argument("--wav", required=True, help="Path to an audio file for testing")
    parser.add_argument("--output", default="./test_output", help="Output directory for test results")
    
    args = parser.parse_args()
    
    # Test the vocoder
    output_wav = test_vocoder(args.wav, args.output)
    
    # Test the speaker encoder using the original and reconstructed waveforms
    orig_embeddings = test_speaker_encoder(args.wav)
    recon_embeddings = test_speaker_encoder(output_wav)
    
    # Compare the embeddings to see if speaker identity is preserved
    cosine_sim = torch.nn.functional.cosine_similarity(
        orig_embeddings[0].cpu(), recon_embeddings[0].cpu()
    )
    
    print(f"\nCosine similarity between original and reconstructed embeddings: {cosine_sim.item():.4f}")
    print("A value close to 1.0 indicates the speaker identity was preserved well")
    
    print("\nTests completed successfully!")


if __name__ == "__main__":
    main()
