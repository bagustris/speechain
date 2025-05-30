#!/usr/bin/env python3
import os
import sys
import subprocess

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set environment variables
os.environ["SPEECHAIN_ROOT"] = os.path.abspath(os.path.dirname(__file__))
os.environ["SPEECHAIN_PYTHON"] = sys.executable

# Run the data_dumping.sh script with the correct arguments
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
subprocess.run(cmd, check=True)
