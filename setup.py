from setuptools import find_packages, setup

# Note: SpeeChain no longer depends on SpeechBrain as of v0.2
# Custom implementation of vocoder and speaker embedding are now provided
setup(name="speechain",
      version="0.2",
      description="The main folder of the SpeeChain toolkit.",
      author="Heli Qi (original codes are from Andros Tjandra & Sashi Novitasari)",
      author_email='qi.heli.qi9@is.naist.jp',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="BSD",
      url="",
      packages=find_packages(include=['speechain', 'datasets']),
      install_requires=['numpy',
                        'scipy',
                        'torch',
                        'pytest',
                        'tabulate',
                        'tqdm',
                        'pathos',
                        'librosa',
                        'tensorboardX',
                        'pandas',
                        'tables',
                        'python-speech-features',
                        'soundfile',
                        'psutil',
                        'tensorboard',
                        'pyyaml',
                        'torchaudio',
                        'sentencepiece'])

