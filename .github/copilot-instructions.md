# Copilot Instructions for SpeeChain

## Repository Summary

SpeeChain is a PyTorch-based machine speech chain toolkit for Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) synthesis. It supports multi-GPU training, real-time visualization, on-the-fly data processing, and flexible model training pipelines.

**Languages/Frameworks:** Python 3.10+, PyTorch, CUDA
**Project Type:** Python ML toolkit with pip package
**Size:** ~100 Python source files in the `speechain/` module

## Build and Validation Commands

### Environment Setup
1. Install the package in development mode:
   ```bash
   uv pip install -e .
   ```
2. This installs the `speechain` package from `setup.py` with all dependencies.

### Linting (ALWAYS run before committing)
The repository uses **Black** for formatting and **Ruff** for import sorting. CI workflows enforce these.

1. **Format check with Black:**
   ```bash
   black --check speechain/
   ```
2. **Fix formatting with Black:**
   ```bash
   black speechain/
   ```
3. **Check import sorting with Ruff:**
   ```bash
   ruff check --select I speechain/
   ```
4. **Fix import sorting with Ruff:**
   ```bash
   ruff check --select I speechain/ --fix
   ```

### Testing
Run pytest on test files:
```bash
python -m pytest speechain/utilbox/test_humanfriendly.py -v
```

Test files follow the pattern `test_*.py` in the source tree:
- `speechain/utilbox/test_humanfriendly.py`
- `speechain/module/encoder/test_speaker.py`
- `speechain/module/vocoder/test_hifigan.py`

### Verify Package Import
```bash
python -c "import speechain; print(speechain)"
```

## Project Layout

### Root Directory Structure
```
speechain/               # Main Python package (source code)
recipes/                 # Task-specific experiment configurations
datasets/                # Dataset processing scripts and metadata
config/                  # Shared configuration files
docs/                    # Documentation (mkdocs)
.github/workflows/       # CI workflows (Black, Ruff, Documentation)
setup.py                 # Package installation
environment.yaml         # Conda environment specification
requirements.txt         # Pip dependencies
create_env.sh            # Environment setup script
mkdocs.yml               # Documentation configuration
```

### Main Package Structure (`speechain/`)
```
speechain/
├── runner.py            # Main entry point for training/testing
├── monitor.py           # Training/testing monitors
├── snapshooter.py       # Figure generation for visualization
├── criterion/           # Loss functions
├── dataset/             # Data loading classes
├── infer_func/          # Inference functions (beam search, TTS decoding)
├── iterator/            # Data batching iterators
├── model/               # Model definitions (ASR, TTS, LM)
├── module/              # Neural network modules (transformer, conformer, etc.)
├── optim_sche/          # Optimizer and scheduler classes
├── tokenizer/           # Text tokenization classes
├── utilbox/             # Utility functions
└── pyscripts/           # Additional Python scripts
```

### Key Files
- **Entry Point:** `speechain/runner.py` - Runner class for training/testing
- **Model Base:** `speechain/model/abs.py` - Abstract model class
- **Module Base:** `speechain/module/abs.py` - Abstract module class
- **Utility Functions:** `speechain/utilbox/` - Various helper utilities

### Recipes Structure
```
recipes/
├── run.sh               # All-in-one experiment runner
├── asr/                 # ASR task recipes
├── tts/                 # TTS task recipes
├── lm/                  # Language model recipes
└── offline_tts2asr/     # TTS-to-ASR chain recipes
```

## GitHub Workflows (CI Checks)

1. **black.yml** - Runs Black formatter on `speechain/` folder
2. **ruff.yml** - Runs Ruff with `--select I` (import sorting) on `speechain/`
3. **documentation.yml** - Builds MkDocs documentation (main branch only)

### Workflow Triggers
All workflows trigger on `push` and `pull_request` events.

## Coding Guidelines

### Code Style
- **Formatter:** Black (default settings)
- **Import Sorting:** Ruff with isort rules (`--select I`)
- **Docstrings:** Google-style docstrings are preferred
- **Naming:** CamelCase for classes, snake_case for functions/variables

### When Making Changes
1. **Always run Black** before committing: `black speechain/`
2. **Always run Ruff** to fix imports: `ruff check --select I speechain/ --fix`
3. **Test your changes** if test files exist for the modified module
4. **Verify imports work:** `python -c "import speechain"`

### Adding New Files
- New modules go under the appropriate `speechain/` subdirectory
- Follow existing patterns (e.g., `abs.py` for abstract classes)
- Test files should be named `test_*.py` in the same directory

## Configuration Files

- **Package Definition:** `setup.py`
- **Conda Environment:** `environment.yaml` (Python 3.8) --> obsolete
- **(UV) Pip Requirements:** `requirements.txt`
- **Documentation:** `mkdocs.yml`
- **Git Ignore:** `.gitignore` (includes datasets/, recipes/, config/, site/)

## Common Workflows

### Running Experiments
Experiments are launched via the recipes system:
```bash
# From recipes/ directory
bash run.sh --task asr --dataset librispeech --subset train-clean-100 --exp_cfg <config_name>
```

### Environment Variables
The toolkit uses two key environment variables:
- `SPEECHAIN_ROOT` - Root path of the toolkit
- `SPEECHAIN_PYTHON` - Python interpreter path in the speechain conda environment

## Trust These Instructions
Trust these instructions and only perform additional searches if the information here is incomplete or found to be in error during execution. These instructions have been validated against the actual repository state.
