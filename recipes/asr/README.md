# ASR

ASR (Automatic Speech Recognition) is a technology that converts spoken language into text. ASR is widely used in various applications such as voice assistants, dictation software, and transcription services. SpeeChain provides a collection of ASR recipes that allow users to train and evaluate ASR models on various datasets. The recipes include configurations for different backbones, such as transformer and conformer, and provide pretrained models for reproducibility. Users can also create their own ASR models by following the instructions in the recipes.

👆[Back to the recipe README.md](https://github.com/bagustris/SpeeChain/tree/main/recipes#recipes-folder-of-the-speechain-toolkit)

## Table of Contents
1. [Available Backbones](https://github.com/bagustris/SpeeChain/tree/main/recipes/asr#available-backbones)
2. [Pretrained Models for Reproducibility](https://github.com/bagustris/SpeeChain/tree/main/recipes/asr#pretrained-models-for-reproducibility)
3. [Training an ASR model](https://github.com/bagustris/SpeeChain/tree/main/recipes/asr#training-an-asr-model)
4. [Creating your own ASR model](https://github.com/bagustris/SpeeChain/tree/main/recipes/asr#creating-your-own-asr-model)

## Available Backbones (WER with/without LM)
<table>
	<tr>
	    <th>Dataset</th>
	    <th>Subset</th>
	    <th>Configuration</th>
	    <th>Test Clean</th>  
	    <th>Test Other</th>  
	</tr>
   <tr>
      <td rowspan="9">librispeech</td>
      <td rowspan="1">train-clean-5</td>
      <td>5-bpe5k_conformer-small_lr2e-3</td>
      <td>15.2% / 35.8%</td>
      <td>35.8% / 42.3%</td>
   </tr>
	<tr>
	    <!-- <td rowspan="8">librispeech</td> -->
	    <td rowspan="4">train-clean-100</td>
	    <td>100-bpe5k_transformer-wide_lr2e-3</td>
	    <td> 8.40% / 21.92% </td>
	    <td> 5.50% / 15.56% </td>
	</tr>
    <tr>
        <td>100-bpe5k_conformer-small_lr2e-3</td>
	    <td> 6.3% / 9.3% </td>
	    <td> 19.14% / 25.2% </td>
    </tr>
    <tr>
        <td>100-bpe5k_conformer-medium_lr2e-3</td>
	    <td> 7.87% / 21.36% </td>
	    <td> 5.30% / 15.57% </td>
    </tr>
    <tr>
        <td>100-bpe5k_conformer-large_lr2e-3</td>
	    <td> 7.30% / 20.24% </td>
	    <td> 5.33% / 15.15% </td>
    </tr>
	<tr>
	    <td rowspan="2">train-clean-460</td>
        <td>460-bpe5k_transformer-large</td>
	    <td> % / % </td>
        <td> % / % </td>
	</tr>
    <tr>
        <td>460-bpe5k_conformer-large</td>
	    <td> % / % </td>
	    <td> % / % </td>
    </tr>
	<tr>
	    <td rowspan="2">train-960</td>
	    <td>960-bpe5k_transformer-large</td>
	    <td> % / % </td>
        <td> % / % </td>
	</tr>
    <tr>
        <td>960-bpe5k_conformer-large</td>
	    <td> % / % </td>
        <td> % / % </td>
    </tr>
    <tr>
	    <td rowspan="1">libritts_librispeech</td>
	    <td rowspan="1">train-960</td>
	    <td>960-bpe5k_transformer-large</td>
	    <td> % / % </td>
	    <td> % / % </td>
	</tr>
</table>

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain/tree/main/recipes/asr#table-of-contents)


## Pretrained Models for Reproducibility
For reproducibility of our ASR model configuration files in `${SPEECHAIN_ROOT}/recipes/asr/`, we provide the following pretrained models to ensure consistent performance:  

1. SentencePiece tokenizer models  

    * Please download tokenizer model and vocabulary to where your dataset is dumped. The default path is `${SPEECHAIN_ROOT}/datasets`.   
    **Note:** If your dataset is dumped outside SpeeChain, please replace `${SPEECHAIN_ROOT}/datasets` in the following commands by your place.
     
    * **LibriSpeech:**  

         1. **train-clean-100:**  

            ```bash
            # Download BPE model
            gdown -O ${SPEECHAIN_ROOT}/datasets/librispeech/data/sentencepiece/train-clean-100/bpe5k/no-punc 

            # Download BPE vocabulary
            gdown -O ${SPEECHAIN_ROOT}/datasets/librispeech/data/sentencepiece/train-clean-100/bpe5k/no-punc 
            ```

         2. **train-clean-460:**  

            ```bash
            # Download BPE model
            gdown -O ${SPEECHAIN_ROOT}/datasets/librispeech/data/sentencepiece/train-clean-100/bpe5k/no-punc 

            # Download BPE vocabulary
            gdown -O ${SPEECHAIN_ROOT}/datasets/librispeech/data/sentencepiece/train-clean-100/bpe5k/no-punc 
            ```

         3. **train-960:** 
            
            ```bash
            # Download BPE model
            gdown -O ${SPEECHAIN_ROOT}/datasets/librispeech/data/sentencepiece/train-clean-100/bpe5k/no-punc 

            # Download BPE vocabulary by 
            gdown -O ${SPEECHAIN_ROOT}/datasets/librispeech/data/sentencepiece/train-clean-100/bpe5k/no-punc 
            ```

3. Transformer-based language models  

    * Please download both LM model and configuration file. The default path is `${SPEECHAIN_ROOT}/recipes/lm`.   
    **Note:** If you want to store model files outside SpeeChain, please replace `${SPEECHAIN_ROOT}/recipes/lm` in the following commands by your place. Also, change the `lm_cfg_path` and `lm_model_path` arguments in each ASR configuration file.
    * **LibriSpeech:**  
    
         1. **train-clean-100:** 
               
               ```bash  
               # Download LM model  
               gdown -O ${SPEECHAIN_ROOT}/recipes/lm/librispeech/lm_text/exp/100-bpe5k_transformer_gelu/models   
               # Download LM configuration  
               gdown -O ${SPEECHAIN_ROOT}/recipes/lm/librispeech/lm_text/exp/100-bpe5k_transformer_gelu  
               ```
      
         2. **train-960:**
               
               ```bash  
               # Download LM model
               gdown -O ${SPEECHAIN_ROOT}/recipes/lm/librispeech/train-960_lm_text/exp/960-bpe5k_transformer_gelu/models 
      
               # Download LM configuration
               gdown -O ${SPEECHAIN_ROOT}/recipes/lm/librispeech/train-960_lm_text/exp/960-bpe5k_transformer_gelu  
               ```

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain/tree/main/recipes/asr#table-of-contents)


## Training an ASR model  
Before training an ASR model, ensure that your target datasets are dumped by the scripts in `${SPEECHAIN_ROOT}/datasets/{your-target-dataset}`.
More details on how to dump a dataset can be found [here](https://github.com/bagustris/SpeeChain/tree/main/datasets#how-to-dump-a-dataset-on-your-machine).

### Use an existing dataset with a pre-tuned configuration
1. locate a _.yaml_ configuration file in `${SPEECHAIN_ROOT}/recipes/asr`. 
   Suppose we want to train an ASR model by the configuration `${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp_cfg/960-bpe5k_transformer-wide_ctc_perturb.yaml`.

2. Train and evaluate the ASR model on your target training set
   
   ```bash
   cd ${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960
   bash run.sh --exp_cfg 960-bpe5k_transformer-wide_ctc_perturb (--ngpu x --gpus x,x)
   ```
   **Note:**   
      1. Review the comments on the top of the configuration file to ensure that your computational resources fit the configuration before training the model.  
      If your resources do not match the configuration, adjust it by `--ngpu` and `--gpus` to match your available GPU memory.  
      2. To save the experimental results outside the toolkit folder `${SPEECHAIN_ROOT}`, 
         specify your desired location by appending `--train_result_path {your-target-path}` to `bash run.sh`.  
         In this example, `bash run.sh --exp_cfg 960-bpe5k_transformer-wide_ctc_perturb --train_result_path /a/b/c`
         will save results to `/a/b/c/960-bpe5k_transformer-wide_ctc_perturb`.

### Creating a new configuration for a non-existing dataset  

1. Dump your target dataset from the Internet following these [instructions](https://github.com/bagustris/SpeeChain/tree/main/datasets#how-to-contribute-a-new-dataset).  

2. Create a folder for your dumped dataset `${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}`:
   ```bash  
   mkdir ${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}
   cp ${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/run.sh ${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}/run.sh
   ```  
   **Note:**  
      - Update the arguments `dataset` and `subset` (line no.16 & 17) in `${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}/run.sh`:  

      ```
      dataset=librispeech -> 'your-new-dataset'
      subset='train-960' -> 'your-new-dataset'
      ```

3. Copy a pre-tuned configuration file into your newly created folder. 
   Suppose we want to use the configuration `${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp_cfg/960-bpe5k_transformer-wide_ctc_perturb.yaml`:  
   ```bash  
   cd ${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}
   mkdir ./data_cfg ./exp_cfg
   cp ${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/exp_cfg/960-bpe5k_transformer-wide_ctc_perturb.yaml ./exp_cfg
   ```
   **Note:**  
      - Update the dataset arguments at the beginning of your selected configuration:
      ```  
      # dataset-related
      dataset: librispeech -> 'your-new-dataset'
      train_set: train-960 -> 'your-target-subset'
      valid_set: dev -> 'valid-set-of-new-dataset'
      
      # tokenizer-related
      txt_format: asr
      vocab_set: train-960 -> 'your-target-subset'
      token_type: sentencepiece
      token_num: bpe5k
      ```

4. Train the ASR model on your target training set:
   ```bash  
   cd ${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}
   bash run.sh --test false --exp_cfg 960-bpe5k_transformer-wide_ctc_perturb (--ngpu x --gpus x,x)
   ```
   **Note:**  
      1. `--test false` is used to skip the testing stage.
      2. Ensure your computational resources match the configuration before training the model.
      3. To save experimental results outside ${SPEECHAIN_ROOT}, specify your desired location by appending --train_result_path {your-target-path} to bash run.sh.

5. Tune the inference hyperparameters on the corresponding validation set
   ```bash  
   cp ${SPEECHAIN_ROOT}/recipes/asr/librispeech/train-960/data_cfg/test_dev-clean+other.yaml ./data_cfg
   mv ./data_cfg/test_dev-clean.yaml ./data_cfg/test_{your-valid-set-name}.yaml
   bash run.sh --train false --exp_cfg 960-bpe5k_transformer-wide_ctc_perturb --data_cfg test_{your-valid-set-name}
   ```
   **Note:**  
      1. Update the dataset arguments in `./data_cfg/test_{your-valid-set-name}.yaml`:
         ```  
         dataset: librispeech -> 'your-new-dataset'
         valid_dset: &valid_dset dev-clean -> &valid_dset 'valid-set-of-new-dataset'
         ```

      2. `--train false` is used to skip the training stage.
      3. `--data_cfg` switches the data loading configuration from the original one for training in exp_cfg to the one for validation tuning.
      4. To access experimental results saved outside `${SPEECHAIN_ROOT}`, append `--train_result_path {your-target-path}` to `bash run.sh`.

6. Evaluate the trained ASR model on the official test sets
   ```bash
   bash run.sh --train false --exp_cfg 960-bpe5k_transformer-wide_ctc_perturb --infer_cfg "{the-best-configuration-you-get-during-validation-tuning}"
   ```
   **Note:**  
      1. `--train false` is used to skip the training stage.  
      2. There are two ways to specify the optimal `infer_cfg` tuned on the validation set:  
         1. Update `infer_cfg` in `${SPEECHAIN_ROOT}/recipes/asr/{your-new-dataset}/{your-target-subset}/exp_cfg/960-bpe5k_transformer-wide_ctc_perturb.yaml`.  
         2. Provide a parsable string as the value for `--infer_cfg` in the terminal. For example, `beam_size:16,ctc_weight:0.2` can be converted into a dictionary with two key-value items (`beam_size=16` and `ctc_weight=0.2`).  
         For more details about this syntax, refer to [**here**](https://github.com/bagustris/SpeeChain/blob/main/handbook.md#convertable-arguments-in-the-terminal).  
      3. To access experimental results saved outside `${SPEECHAIN_ROOT}`, append `--train_result_path {your-target-path}` to `bash run.sh`.

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain/tree/main/recipes/asr#table-of-contents)

# How to create your own ASR model
The detailed instructions for creating your own ASR model using SpeeChain are coming soon.

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain/tree/main/recipes/asr#table-of-contents)
