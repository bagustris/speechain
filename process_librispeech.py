#!/usr/bin/env python3
"""
Comprehensive wrapper script for processing LibriSpeech data in SpeeChain.
This script handles Python path setup and dependency installation.
"""
import os
import sys
import subprocess
import importlib.util

# Add the project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Set environment variables
os.environ["SPEECHAIN_ROOT"] = project_root
os.environ["SPEECHAIN_PYTHON"] = sys.executable

# Check and install required dependencies
required_packages = ["h5py", "g2p_en", "GPUtil", "sentencepiece", "tqdm", "numpy", "torch"]

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in required_packages:
    try:
        importlib.import_module(package)
        print(f"{package} is already installed.")
    except ImportError:
        install_package(package)

# Create __init__.py files in datasets directory and subdirectories if they don't exist
for root, dirs, files in os.walk(os.path.join(project_root, "datasets")):
    if "__init__.py" not in files:
        init_path = os.path.join(root, "__init__.py")
        with open(init_path, "w") as f:
            pass
        print(f"Created {init_path}")

# Download and process LibriSpeech data
def run_data_processing():
    cmd = [
        "bash", "datasets/data_dumping.sh",
        "--dataset_name", "librispeech",
        "--subsets", "train-clean-5 dev-clean-2",
        "--vocab_src_subsets", "train-clean-5",
        "--token_type", "sentencepiece",
        "--vocab_generate_args", "--vocab_size 1000 --model_type bpe",
        "--download_args", "--subsets train-clean-5,dev-clean-2",
        "--meta_generate_args", "",
        "--meta_post_process_args", ""
    ]
    print("Running command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print("Data processing completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Data processing failed with error code {e.returncode}")
        print("Trying to run individual steps manually...")
        run_manual_processing()

def run_manual_processing():
    """Run individual processing steps manually if the script fails"""
    # Step 1: Download data if needed
    if not os.path.exists(os.path.join(project_root, "datasets/librispeech/data/wav/train-clean-5")):
        subprocess.run(["bash", "datasets/librispeech/data_download.sh", 
                       "--download_path", os.path.join(project_root, "datasets/librispeech/data"),
                       "--subsets", "train-clean-5,dev-clean-2"], check=True)
    
    # Step 2: Generate metadata
    try:
        # Create a temporary script to run the meta generator
        with open("temp_meta_generator.py", "w") as f:
            f.write("""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from datasets.librispeech.meta_generator import LibriSpeechMetaGenerator

if __name__ == "__main__":
    generator = LibriSpeechMetaGenerator()
    generator.generate_meta_dict(
        src_path=os.path.join(os.environ["SPEECHAIN_ROOT"], "datasets/librispeech/data/wav"),
        txt_format="no-punc",
        subsets="train-clean-5,dev-clean-2",
        separator=",",
        ncpu=8
    )
""")
        subprocess.run([sys.executable, "temp_meta_generator.py"], check=True)
    except Exception as e:
        print(f"Metadata generation failed: {e}")
    
    # Step 3: Generate sentencepiece model
    try:
        # Create a temporary script to run the vocab generator
        with open("temp_vocab_generator.py", "w") as f:
            f.write("""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from datasets.pyscripts.vocab_generator import main

if __name__ == "__main__":
    text_path = os.path.join(os.environ["SPEECHAIN_ROOT"], "datasets/librispeech/data/wav/train-clean-5/idx2no-punc_text")
    save_path = os.path.join(os.environ["SPEECHAIN_ROOT"], "datasets/librispeech/data/sentencepiece/train-clean-5/bpe1k/no-punc")
    os.makedirs(save_path, exist_ok=True)
    main(
        text_path=text_path,
        save_path=save_path,
        token_type="sentencepiece",
        txt_format="no-punc",
        vocab_size=1000,
        model_type="bpe",
        character_coverage=1.0,
        split_by_whitespace=True
    )
""")
        subprocess.run([sys.executable, "temp_vocab_generator.py"], check=True)
    except Exception as e:
        print(f"Vocab generation failed: {e}")

if __name__ == "__main__":
    run_data_processing()
