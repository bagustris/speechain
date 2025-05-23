# Changelog

All notable changes to this project will be documented in this file.

## [unreleased]

## [0.2.0] - 2025-05-23 

### ✨ New Features

- Removed SpeechBrain dependency
- Added custom implementation of HiFiGAN vocoder
- Added custom implementation of speaker embedding models
- Added test script for vocoder and speaker embedding

### 🐛 Bug Fixes

- Bugs in metadata_generator.py librispeech
- Conflict in lm_text/exp_cfg/100-*
- Linting

### 💼 Other

- __init__.py

### 📚 Documentation

- Fix format
- Update
- Update tts

### 🎨 Styling

- Formatted with black

### ⚙️ Miscellaneous Tasks

- Add inverted logo
- Update gitignore
- Update gitignore
- Clean repo
- Clean repo
- Replace humanfriendly package with python code, add test
- Clean requirements
- Update CHANGELOG for version 0.1.2, add new requirements.txt and humanfriendly.py
- Update docs

## [0.1.2] - 2024-12-11 

### Added
- a new requirements.txt file
- `humanfriendly.py` in `utilbox` along with its test file


## [0.1.1] - 2024-09-30

### Added
- yaml file is added `recipes/asr/librispeech/train-960/exp_cfg/960-bpe5k_transformer-wide_ctc_perturb.yaml`
- train-clean-5 recipe

### Changed
- repo name from SpeeChain to speechain
- bacth size to 2.4e7 (from 2.4e7) in yaml file above.
- `envir_preparation.sh` to `create_env.sh`

### Fixed
- Error in `meta_generator.py` for librispeech dataset


## [0.1.0] - 2024-09-26 
- Forked from `heli-qi/SpeeChain