#!/usr/bin/env python3
"""
Fix metadata generation for LibriSpeech dataset.
This script addresses import issues and ensures proper creation of idx2wav and idx2no-punc_text files.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

SPEECHAIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
os.environ["SPEECHAIN_ROOT"] = SPEECHAIN_ROOT
os.environ["SPEECHAIN_PYTHON"] = sys.executable

sys.path.insert(0, os.path.join(SPEECHAIN_ROOT, "datasets"))

def ensure_dirs():
    """Create necessary directories for metadata files"""
    dirs = [
        os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav/train-clean-5"),
        os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav/dev-clean-2"),
        os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/sentencepiece/train-clean-5/bpe1k/no-punc")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    src_train = os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/data/wav/train-clean-5")
    src_dev = os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/data/wav/dev-clean-2")
    
    tgt_train = os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav/train-clean-5")
    tgt_dev = os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav/dev-clean-2")
    
    if os.path.exists(src_train) and not os.listdir(tgt_train):
        print(f"Creating symlink from {src_train} to {tgt_train}")
        for item in os.listdir(src_train):
            src_item = os.path.join(src_train, item)
            tgt_item = os.path.join(tgt_train, item)
            if not os.path.exists(tgt_item):
                os.symlink(src_item, tgt_item)
    
    if os.path.exists(src_dev) and not os.listdir(tgt_dev):
        print(f"Creating symlink from {src_dev} to {tgt_dev}")
        for item in os.listdir(src_dev):
            src_item = os.path.join(src_dev, item)
            tgt_item = os.path.join(tgt_dev, item)
            if not os.path.exists(tgt_item):
                os.symlink(src_item, tgt_item)
    
    init_files = [
        os.path.join(SPEECHAIN_ROOT, "datasets/__init__.py"),
        os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/__init__.py"),
        os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/__init__.py"),
        os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav/__init__.py"),
        os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav/train-clean-5/__init__.py"),
        os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav/dev-clean-2/__init__.py")
    ]
    for f in init_files:
        if not os.path.exists(f):
            with open(f, 'w') as file:
                pass

def run_command(cmd, cwd=None):
    """Run a shell command and return the output"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=cwd, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def generate_metadata():
    """Generate metadata files for LibriSpeech dataset"""
    meta_generator_path = os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/meta_generator.py")
    meta_post_processor_path = os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/meta_post_processor.py")
    
    cmd = f"{sys.executable} {meta_generator_path} --tgt_path {SPEECHAIN_ROOT}/datasets/librispeech/data/wav --txt_format no-punc"
    success = run_command(cmd)
    if not success:
        print("Metadata generation failed")
        return False
    
    cmd = f"{sys.executable} {meta_post_processor_path} --src_path {SPEECHAIN_ROOT}/datasets/librispeech/data/wav"
    success = run_command(cmd)
    if not success:
        print("Metadata post-processing failed")
        return False
    
    return True

def manually_create_metadata_files():
    """Manually create idx2wav and idx2no-punc_text files if automatic generation fails"""
    train_dir = os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav/train-clean-5")
    dev_dir = os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav/dev-clean-2")
    
    for subset_dir, subset_name in [(train_dir, "train-clean-5"), (dev_dir, "dev-clean-2")]:
        idx2wav_path = os.path.join(subset_dir, "idx2wav")
        if not os.path.exists(idx2wav_path):
            print(f"Manually creating {idx2wav_path}")
            with open(idx2wav_path, 'w') as f:
                base_dir = os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav", subset_name)
                for root, _, files in os.walk(base_dir):
                    for file in files:
                        if file.endswith(".flac"):
                            idx = file.split(".")[0]
                            path = os.path.abspath(os.path.join(root, file))
                            f.write(f"{idx} {path}\n")
    
    for subset_dir, subset_name in [(train_dir, "train-clean-5"), (dev_dir, "dev-clean-2")]:
        idx2text_path = os.path.join(subset_dir, "idx2no-punc_text")
        if not os.path.exists(idx2text_path):
            print(f"Manually creating {idx2text_path}")
            with open(idx2text_path, 'w') as f:
                base_dir = os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav", subset_name)
                for root, _, files in os.walk(base_dir):
                    for file in files:
                        if file.endswith(".txt"):
                            with open(os.path.join(root, file), 'r') as txt_file:
                                for line in txt_file:
                                    parts = line.strip().split(" ", 1)
                                    if len(parts) == 2:
                                        idx, text = parts
                                        processed_text = text.lower()
                                        for char in ",.!?;:\"'()[]{}":
                                            processed_text = processed_text.replace(char, "")
                                        f.write(f"{idx} {processed_text}\n")

def generate_sentencepiece_tokenizer():
    """Generate sentencepiece tokenizer model"""
    vocab_generator_path = os.path.join(SPEECHAIN_ROOT, "datasets/pyscripts/vocab_generator.py")
    
    os.makedirs(os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/sentencepiece/train-clean-5"), exist_ok=True)
    
    idx2text_path = os.path.join(SPEECHAIN_ROOT, "datasets/librispeech/data/wav/train-clean-5/idx2no-punc_text")
    if not os.path.exists(idx2text_path):
        print(f"Error: {idx2text_path} does not exist. Cannot generate tokenizer.")
        return False
    
    cmd = (f"{sys.executable} {vocab_generator_path} "
           f"--text_path {SPEECHAIN_ROOT}/datasets/librispeech/data/wav/train-clean-5 "
           f"--save_path {SPEECHAIN_ROOT}/datasets/librispeech/data/sentencepiece/train-clean-5 "
           f"--token_type sentencepiece "
           f"--txt_format no-punc "
           f"--vocab_size 1000 "
           f"--model_type bpe")
    
    success = run_command(cmd)
    if not success:
        print("Sentencepiece tokenizer generation failed")
        return False
    
    return True

def main():
    """Main function to fix metadata generation"""
    print("Starting metadata fix process...")
    
    ensure_dirs()
    
    if generate_metadata():
        print("Metadata generation successful")
    else:
        print("Falling back to manual metadata creation")
        manually_create_metadata_files()
    
    if generate_sentencepiece_tokenizer():
        print("Sentencepiece tokenizer generation successful")
    else:
        print("Sentencepiece tokenizer generation failed")
    
    print("Metadata fix process completed")

if __name__ == "__main__":
    main()
