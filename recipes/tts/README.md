# TTS

This folder contains recipes for training a Text-To-Speech Synthesis (TTS) model.

👆[Back to the recipe README.md](https://github.com/bagustris/SpeeChain/tree/main/recipes#recipes-folder-of-the-speechain-toolkit)

## Table of Contents
1. [Available Backbones](https://github.com/bagustris/SpeeChain/tree/main/recipes/tts#available-backbones)
2. [Preparing Durations for FastSpeech2](https://github.com/bagustris/SpeeChain/tree/main/recipes/tts#preparing-durations-for-fastspeech2)
3. [Training a TTS model](https://github.com/bagustris/SpeeChain/tree/main/recipes/tts#training-a-tts-model)

## Available Backbones
Below is a table of available backbones:
<table>
	<tr>
	    <th>Dataset</th>
	    <th>Subset</th>
	    <th>Configuration</th>
	    <th>Audio Samples Link</th> 
	</tr>
	<tr>
	    <td rowspan="3">libritts</td>
	    <td>train-clean-100</td>
	    <td></td>
	    <td>  </td>
	</tr>
	<tr>
	    <td>train-clean-460</td>
        <td></td>
	    <td>  </td>
	</tr>
	<tr>
	    <td>train-960</td>
	    <td></td>
	    <td> </td>
	</tr>
    <tr>
	    <td rowspan="2">ljspeech</td>
	    <td>train</td>
	    <td>22.05khz_mfa_fastspeech2</td>
	    <td> 
			<audio controls="controls">
  			<source type="audio/mp3" src="samples/tts/ljspeech/22.05kh_mfa_fastspeech2/LJ001-0001.wav"></source>
			</audio>
		</td>
	</tr>
	<tr>
	    <td>train</td>
	    <td>22.05khz_mfa_fastspeech2_nopunc</td>
	    <td>
			<audio controls="controls">
  			<source type="audio/wav" src="samples/tts/ljspeech/22.05kh_mfa_fastspeech2_punc/LJ001-0001.wav"></source>
			</audio>
		</td>
	</tr>
    <tr>
	    <td rowspan="1">vctk</td>
	    <td></td>
	    <td></td>
	    <td>  </td>
	</tr>
</table>

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain/tree/main/recipes/tts#table-of-contents)

## Preparing Durations for FastSpeech2
For training a FastSpeech2 model, you need to acquire additional duration data for your target dataset. 
Follow these steps:  
 1. Create a virtual environment for MFA: `conda create -n speechain_mfa -c conda-forge montreal-forced-aligner gdown`.  
 2. Activate the `speechain_mfa` environment: `conda activate speechain_mfa`.  
 3. Downsample your target TTS dataset to 16khz. For details, please see [how to dump a dataset on your machine](https://github.com/bagustris/SpeeChain/tree/main/datasets#how-to-dump-a-dataset-on-your-machine).  
 4. By default, MFA package will store all the temporary files to your user directory. If you lack sufficient space, add `export MFA_ROOT_DIR={your-target-directory}` to `~/.bashrc` and run `source ~/.bashrc`.  
 5. Navigate to `${SPEECHAIN_ROOT}/datasets` and run `bash mfa_preparation.sh -h` for help. Then, add appropriate arguments to `bash mfa_preparation.sh` to acquire duration data.  

**Note:** MFA cannot process duration calculations for multiple datasets concurrently on a single machine (or a single node on a cluster). 
Please process each dataset one at a time.

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain/tree/main/recipes/tts#table-of-contents)

## Training an TTS model
To train a TTS model, follow the [ASR model training instructions](https://github.com/bagustris/SpeeChain/tree/main/recipes/asr#training-an-asr-model) located in `recipes/asr`. 
Make sure to replace the folder names and configuration file names from `recipes/asr` with their corresponding names in `recipes/tts`.

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain/tree/main/recipes/tts#table-of-contents)
