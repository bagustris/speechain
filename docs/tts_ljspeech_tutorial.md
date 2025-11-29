# TTS LJSpeech Tutorial: FastSpeech2 Experiments

This tutorial guides you through running Text-to-Speech (TTS) experiments on the LJSpeech dataset using FastSpeech2 with two configurations: **no-punctuation** and **punctuation**.

## Prerequisites

- **GPU**: NVIDIA GPU with at least 8GB VRAM (tested on RTX 3090/5090)
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Storage**: ~20GB free space for dataset and models

## Environment Setup

### Option A: Using uv + venv (Recommended)

This option uses `uv` for fast Python environment management without conda.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the speechain directory
cd /path/to/speechain

# Create virtual environment with Python 3.10
uv venv .venv --python 3.10

# Activate the environment
source .venv/bin/activate

# Install PyTorch (adjust CUDA version as needed)
# For CUDA 12.1:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For newer GPUs (e.g., RTX 5090 with sm_120), use PyTorch nightly:
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install speechain requirements
uv pip install -r requirements.txt

# Install speechain in development mode
uv pip install -e .
```

### Option B: Using Conda

```bash
# Create conda environment
conda create -n speechain python=3.10 -y
conda activate speechain

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt

# Install speechain
pip install -e .
```

## Step 1: Download LJSpeech Dataset

```bash
# Set environment variable
export SPEECHAIN_ROOT=/path/to/speechain

# Navigate to dataset directory
cd $SPEECHAIN_ROOT/datasets/ljspeech

# Download LJSpeech dataset
bash data_download.sh
```

This downloads and extracts the LJSpeech dataset (~2.6GB) to `datasets/ljspeech/data/`.

## Step 2: Install Montreal Forced Aligner (MFA)

MFA is required for phoneme alignment. **This step requires conda** even if you're using venv for the main environment.

```bash
# Create a separate conda environment for MFA
conda create -n aligner -c conda-forge montreal-forced-aligner -y

# Activate MFA environment
conda activate aligner

# Download MFA models
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

## Step 3: Generate Metadata

```bash
# Activate your main environment (venv or conda)
source $SPEECHAIN_ROOT/.venv/bin/activate  # or: conda activate speechain

# Set environment
export SPEECHAIN_ROOT=/path/to/speechain
cd $SPEECHAIN_ROOT/datasets/ljspeech

# Generate metadata for train/valid/test splits
python meta_generator.py
```

## Step 4: Run MFA Alignment

**Important**: This step must be run in the conda MFA environment.

```bash
# Activate MFA environment
conda activate aligner

# Navigate to dataset preparation scripts
cd $SPEECHAIN_ROOT/datasets

# Prepare data for MFA alignment
bash mfa_preparation.sh ljspeech

# Run MFA alignment (this may take 30-60 minutes)
# The script aligns phonemes to audio using the english_us_arpa model
mfa align \
    $SPEECHAIN_ROOT/datasets/ljspeech/data/mfa_input \
    english_us_arpa \
    english_us_arpa \
    $SPEECHAIN_ROOT/datasets/ljspeech/data/mfa/acoustic=english_us_arpa_lexicon=english_us_arpa \
    --clean
```

## Step 5: Post-process MFA Output

```bash
# Switch back to main environment
source $SPEECHAIN_ROOT/.venv/bin/activate  # or: conda activate speechain

cd $SPEECHAIN_ROOT/datasets/ljspeech

# Generate duration files from MFA alignment
python meta_post_processor.py
```

## Step 6: Generate Duration Data

```bash
cd $SPEECHAIN_ROOT/datasets/pyscripts

# Generate duration files for training
python duration_calculator.py \
    --data_root $SPEECHAIN_ROOT/datasets/ljspeech/data \
    --mfa_model acoustic=english_us_arpa_lexicon=english_us_arpa
```

## Step 7: Training

### Train No-Punctuation Model

```bash
# Activate main environment
source $SPEECHAIN_ROOT/.venv/bin/activate
export SPEECHAIN_ROOT=/path/to/speechain

cd $SPEECHAIN_ROOT

# Run training (adjust num_epochs in config for full training)
python speechain/runner.py \
    --config recipes/tts/ljspeech/exp_cfg/22.05khz_mfa_fastspeech2.yaml \
    --train true \
    --test false
```

### Train Punctuation Model

```bash
python speechain/runner.py \
    --config recipes/tts/ljspeech/exp_cfg/22.05khz_mfa_fastspeech2_punc.yaml \
    --train true \
    --test false
```

### Training Configuration

Key parameters in the config files (`recipes/tts/ljspeech/exp_cfg/`):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_epochs` | Number of training epochs | 500 |
| `batch_len` | Batch length for training | 1.5e7 |
| `ngpu` | Number of GPUs | 1 |
| `early_stopping_patience` | Epochs before early stopping | 20 |
| `valid_per_epochs` | Validation frequency | 10 |

For quick experiments, you can reduce `num_epochs` to 5-10.

## Step 8: Inference (Generate Speech)

### Generate WAV Files from No-Punctuation Model

```bash
python speechain/runner.py \
    --config recipes/tts/ljspeech/exp_cfg/22.05khz_mfa_fastspeech2.yaml \
    --train false \
    --test true \
    --test_model latest
```

Output WAV files will be saved to:
```
recipes/tts/ljspeech/exp/22.05khz_mfa_fastspeech2/default_inference/latest/test/wav/
```

### Generate WAV Files from Punctuation Model

```bash
python speechain/runner.py \
    --config recipes/tts/ljspeech/exp_cfg/22.05khz_mfa_fastspeech2_punc.yaml \
    --train false \
    --test true \
    --test_model latest
```

Output WAV files will be saved to:
```
recipes/tts/ljspeech/exp/22.05khz_mfa_fastspeech2_punc/default_inference/latest/test/wav/
```

## HiFi-GAN Vocoder

The HiFi-GAN vocoder converts mel spectrograms to audio waveforms. It is **automatically downloaded** from HuggingFace Hub on first use:

- Model: `speechbrain/tts-hifigan-ljspeech`
- Cache location: `recipes/tts/speechbrain_vocoder/hifigan-ljspeech/`

No manual download is required.

## Output Structure

After training and inference, the experiment folder structure looks like:

```
recipes/tts/ljspeech/exp/
├── 22.05khz_mfa_fastspeech2/          # No-punctuation experiment
│   ├── models/                         # Saved model checkpoints
│   │   ├── epoch_1.pth
│   │   ├── epoch_2.pth
│   │   ├── ...
│   │   └── latest.pth -> epoch_N.pth
│   ├── tensorboard/                    # Training logs
│   ├── figures/                        # Visualization
│   ├── default_inference/
│   │   └── latest/
│   │       └── test/
│   │           └── wav/                # Generated WAV files
│   └── train.log
│
└── 22.05khz_mfa_fastspeech2_punc/      # Punctuation experiment
    ├── models/
    ├── tensorboard/
    ├── figures/
    ├── default_inference/
    │   └── latest/
    │       └── test/
    │           └── wav/                # Generated WAV files
    └── train.log
```

## Troubleshooting

### CUDA Out of Memory

Reduce `batch_len` in the config file:
```yaml
batch_len: 1.0e7  # Reduce from 1.5e7
```

### MFA Alignment Errors

Ensure you're using the conda `aligner` environment:
```bash
conda activate aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

### Missing SPEECHAIN_ROOT

Always set the environment variable before running:
```bash
export SPEECHAIN_ROOT=/path/to/speechain
```

### PyTorch CUDA Compatibility

For newer GPUs (RTX 40xx, 50xx), you may need PyTorch nightly:
```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Quick Reference

| Step | Environment | Command |
|------|-------------|---------|
| 1. Setup | venv/conda | `uv venv .venv --python 3.10` |
| 2. Download data | venv/conda | `bash data_download.sh` |
| 3. Install MFA | conda only | `conda create -n aligner -c conda-forge montreal-forced-aligner` |
| 4. Generate meta | venv/conda | `python meta_generator.py` |
| 5. MFA alignment | conda (aligner) | `mfa align ...` |
| 6. Post-process | venv/conda | `python meta_post_processor.py` |
| 7. Training | venv/conda | `python speechain/runner.py --train true` |
| 8. Inference | venv/conda | `python speechain/runner.py --test true` |

## Expected Results

After training for 5 epochs (quick experiment):
- Training loss: ~2.5-3.0
- Generated WAV files: 523 samples per experiment
- Audio quality: Intelligible but may have artifacts (more epochs improve quality)

For production-quality speech, train for 200+ epochs or until early stopping.

## References

- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [FastSpeech2 Paper](https://arxiv.org/abs/2006.04558)
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
- [HiFi-GAN Paper](https://arxiv.org/abs/2010.05646)
