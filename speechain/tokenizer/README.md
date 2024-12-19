# Tokenizer
[_Tokenizer_](https://github.com/bagustris/SpeeChain/blob/main/speechain/tokenizer/abs.py#L14) is the base class of all the _Tokenizer_ objects in this toolkit. 
It on-the-fly transforms text data between strings and tensors.  

For data storage and visualization, the text data should be in the form of strings which is not friendly for model forward calculation. 
For model forward calculation, the text data is better to be in the form of vectors (`torch.tensor` or `numpy.ndarray`).


👆[Back to the handbook page](https://github.com/bagustris/SpeeChain/blob/main/handbook.md#speechain-handbook)

## Table of Contents
1. [**Tokenizer Library**](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#tokenizer-library)
2. [**API Documents**](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#api-document)


## Tokenizer Library
```
/speechain
    /tokenizer
        /abs.py         # Abstract class of Tokenizer. Base of all Tokenizer implementations.
        /char.py        # Tokenizer implementation of the character tokenizer.
        /sp.py          # Tokenizer implementation of the subword tokenizer by SentencePiece package.
        /g2p.py         # Tokenizer implementation of the phoneme tokenizer by G2P package.
```

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#table-of-contents)

## API Document
_Non-overridable backbone functions:_
1. [speechain.tokenizer.abs.Tokenizer.\_\_init__](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#speechaintokenizerabstokenizer__init__self-token_vocab-tokenizer_conf)

_Overridable interface functions:_  
1. [speechain.tokenizer.abs.Tokenizer.tokenizer_init_fn](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#speechaintokenizerabstokenizertokenizer_init_fnself-tokenizer_conf)
2. [speechain.tokenizer.abs.Tokenizer.tensor2text](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#speechaintokenizerabstokenizertensor2textself-tensor)
3. [speechain.tokenizer.abs.Tokenizer.text2tensor](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#speechaintokenizerabstokenizertext2tensorself-text)


### speechain.tokenizer.abs.Tokenizer.\_\_init__(self, token_vocab, **tokenizer_conf)
* **Description:**  
    This function registers some shared member variables for all _Tokenizer_ subclasses: 
    1. `self.idx2token`: the mapping Dict from the token index to token string.
    2. `self.token2idx`: the mapping Dict from the token string to token index.
    3. `self.vocab_size`: the number of tokens in the given vocabulary.
    4. `self.sos_eos_idx`: the index of the joint <sos/eos> token used as the beginning and end of a sentence.
    5. `self.ignore_idx`: the index of the blank token used for either CTC blank modeling or ignored token for encoder-decoder ASR&TTS models.
    6. `self.unk_idx`: the index of the unknown token.
* **Arguments:**
  * _**token_vocab:**_ str  
    The path where the token vocabulary is placed.
  * _****tokenizer_conf:**_  
    The arguments used by `tokenizer_init_fn()` for your customized _Tokenizer_ initialization.

👆[Back to the API list](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#api-document)

### speechain.tokenizer.abs.Tokenizer.tokenizer_init_fn(self, **tokenizer_conf)
* **Description:**  
    This hook interface function initializes the customized part of a _Tokenizer_ subclass if had.  
    This interface is not mandatory to be overridden.
* **Arguments:**
  * _****tokenizer_conf:**_  
    The arguments used by `tokenizer_init_fn()` for your customized _Tokenizer_ initialization.  
    For more details, please refer to the docstring of your target _Tokenizer_ subclass.

👆[Back to the API list](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#api-document)

### speechain.tokenizer.abs.Tokenizer.tensor2text(self, tensor)
* **Description:**  
    This functions decodes a text tensor into a human-friendly string.  
    The default implementation transforms each token index in the input tensor to the token string by `self.idx2token`. 
    If the token index is `self.unk_idx`, an asterisk (*) will be used to represent an unknown token in the string.  
    This interface is not mandatory to be overridden. If your _Tokenizer_ subclass uses some third-party packages to decode the input tensor rather than the built-in `self.idx2token`, 
    please override this function.
* **Arguments:**
  * _**tensor:**_ torch.LongTensor  
    1D integer torch.Tensor that contains the token indices of the sentence to be decoded.
* **Return:**  
    The string of the decoded sentence.

👆[Back to the API list](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#api-document)

### speechain.tokenizer.abs.Tokenizer.text2tensor(self, text)
* **Description:**  
    This functions encodes a text string into a model-friendly tensor.  
    This interface is mandatory to be overridden.  
    By default, this function will attach two <sos/eos> at the beginning and end of the returned token id sequence.
* **Arguments:**
  * _**text:**_ str  
    The input text string to be encoded
  * _**no_sos:**_ bool = False  
    Whether to remove the <sos/eos> at the beginning of the token id sequence.
  * _**no_eos:**_ bool = False  
    Whether to remove the <sos/eos> at the end of the token id sequence.
* **Return:** torch.LongTensor  
    The tensor of the encoded sentence

👆[Back to the API list](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#api-document)

👆[Back to the table of contents](https://github.com/bagustris/SpeeChain/tree/main/speechain/tokenizer#table-of-contents)
