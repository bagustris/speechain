"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.07
"""
import copy
import numpy as np
import torch
from typing import Dict, Any, List

from speechain.model.abs import Model
from speechain.tokenizer.char import CharTokenizer
from speechain.tokenizer.subword import SubwordTokenizer
from speechain.infer_func.beam_search import beam_searching
from speechain.utilbox.tensor_util import to_cpu
from speechain.utilbox.eval_util import get_word_edit_alignment
from speechain.utilbox.md_util import get_list_strings

from speechain.module.encoder.asr import ASREncoder
from speechain.module.decoder.asr import ASRDecoder

from speechain.criterion.cross_entropy import CrossEntropy
from speechain.criterion.accuracy import Accuracy
from speechain.criterion.error_rate import ErrorRate


class ASR(Model):
    """
    Encoder-Decoder Automatic Speech Recognition (Enc-Dec ASR) implementation.

    """
    def model_construction(self,
                           token_type: str,
                           token_vocab: str,
                           frontend: Dict,
                           enc_prenet: Dict,
                           encoder: Dict,
                           dec_prenet: Dict,
                           decoder: Dict,
                           normalize: Dict or bool = None,
                           specaug: Dict or bool = None,
                           cross_entropy: Dict = None,
                           spk_list: str = None,
                           sample_rate: int = 16000,
                           audio_format: str = 'wav'):
        """

        Args:
            # --- module_conf arguments --- #
            (mandatory) frontend:
                The configuration of the acoustic feature extraction frontend.
                This argument must be given since our toolkit doesn't support time-domain ASR.
            (optional) normalize:
                The configuration of the feature normalization module (speechain.module.norm.feat_norm.FeatureNormalization).
                This argument can be given in either a Dict or a bool value.
                In the case of the bool value, True means the default configuration and False means no normalization.
                If this argument is not given, there will be also no normalization.
            (optional) specaug:
                The configuration of the SpecAugment module (speechain.module.augment.specaug.SpecAugment).
                This argument can be given in either a Dict or a bool value.
                In the case of the bool value, True means the default configuration and False means no SpecAugment.
                If this argument is not given, there will be also no SpecAugment.
            (mandatory) enc_prenet:
                The configuration of the prenet in the encoder module.
                The encoder prenet embeds the input acoustic features into hidden embeddings before feeding them into
                the encoder.
            (mandatory) encoder:
                The configuration of the encoder module.
                The encoder embeds the hidden embeddings into the encoder representations at each time steps of the
                input acoustic features.
            (mandatory) dec_prenet:
                The configuration of the prenet in the decoder module.
                The decoder prenet embeds the input token ids into hidden embeddings before feeding them into
                the decoder.
            (mandatory) decoder:
                The configuration of the decoder module.
                The decoder predicts the probability of the next token at each time steps based on the token embeddings.
            # --- criterion_conf arguments --- #
            (optional) cross_entropy:
                The configuration of the cross entropy criterion (speechain.criterion.cross_entropy.CrossEntropy).
                If this argument is not given, the default configuration will be used.
            # --- customize_conf arguments --- #
            (mandatory) token_type:
                The type of the built-in tokenizer.
            (mandatory) token_vocab:
                The absolute path of the vocabulary for the built-in tokenizer.
            (conditionally optional) spk_list:
                The absolute path of the speaker list that contains all the speaker ids.
                If you would like to train a speaker-aware ASR, you need to give a spk_list.
            (optional) sample_rate:
                The sampling rate of the input speech.
                Currently it's used for acoustic feature extraction frontend initialization and tensorboard register of
                the input speech during model visualization.
                In the future, this argument will also be used to dynamically downsample the input speech during training.
            (optional) audio_format:
                The file format of the input speech.
                It's only used for tensorboard register of the input speech during model visualization.

        """
        # --- Model-Customized Part Initialization --- #
        # initialize the tokenizer
        if token_type.lower() == 'char':
            self.tokenizer = CharTokenizer(token_vocab)
        elif token_type.lower() == 'subword':
            # the subword model file is automatically selected in the same folder as the given vocab
            token_model = '/'.join(token_vocab.split('/')[:-1] + ['model'])
            self.tokenizer = SubwordTokenizer(token_vocab, token_model=token_model)
        else:
            raise NotImplementedError

        # initialize the speaker list if given
        if spk_list is not None:
            spk_list = np.loadtxt(spk_list, dtype=str)
            # when the input file is idx2spk, only retain the column of speaker ids
            if len(spk_list.shape) == 2:
                assert spk_list.shape[1] == 2
                spk_list = spk_list[:, 1]
            # otherwise, the input file must be spk_list which is a single-column file and each row is a speaker id
            elif len(spk_list.shape) != 1:
                raise RuntimeError
            # 1. remove redundant elements; 2. sort up the speaker ids in order
            # 3. get the corresponding indices; 4. exchange the positions of indices and speaker ids
            self.spk2idx = dict(map(reversed, enumerate(sorted(set(spk_list.tolist())))))

        # initialize the sampling rate, mainly used for visualizing the input audio during training
        self.sample_rate = sample_rate
        self.audio_format = audio_format.lower()

        # default values of ASR topn bad case selection
        self.bad_cases_selection = [
            ['wer', 'max', 30],
            ['cer', 'max', 30],
            ['deletion', 'max', 30],
            ['insertion', 'max', 30],
            ['substitution', 'max', 30]
        ]


        # --- Module Part Construction --- #
        # Encoder construction, the sampling rate will be first initialized
        if 'sr' not in frontend['conf'].keys():
            frontend['conf']['sr'] = self.sample_rate
        else:
            assert frontend['conf']['sr'] == self.sample_rate, \
                "The sampling rate in your frontend configuration doesn't match the one in customize_conf!"
        self.encoder = ASREncoder(
            frontend=frontend,
            normalize=normalize,
            specaug=specaug,
            prenet=enc_prenet,
            encoder=encoder,
            distributed=self.distributed
        )

        # Decoder construction, the vocabulary size will be first initialized
        if 'vocab_size' in dec_prenet['conf'].keys():
            assert dec_prenet['conf']['vocab_size'] == self.tokenizer.vocab_size, \
                f"The vocab_size values are different in dec_prenet and self.tokenizer! " \
                f"Got dec_prenet['conf']['vocab_size']={dec_prenet['conf']['vocab_size']} and " \
                f"self.tokenizer.vocab_size={self.tokenizer.vocab_size}"
        self.decoder = ASRDecoder(
            vocab_size=self.tokenizer.vocab_size,
            prenet=dec_prenet,
            decoder=decoder
        )


        # --- Criterion Part Initialization --- #
        # training loss
        self.cross_entropy = CrossEntropy(**cross_entropy)
        # validation metrics
        self.accuracy = Accuracy()
        self.error_rate = ErrorRate()


    def batch_preprocess(self, batch_data: Dict):
        """

        Args:
            batch_data:

        Returns:

        """

        def process_strings(data_dict: Dict):
            """
            turn the text strings into tensors and get their lengths

            Args:
                data_dict:

            Returns:

            """
            # --- Process the Text String and its Length --- #
            assert 'text' in data_dict.keys() and isinstance(data_dict['text'], List)
            for i in range(len(data_dict['text'])):
                data_dict['text'][i] = self.tokenizer.text2tensor(data_dict['text'][i])
            text_len = torch.LongTensor([t.size(0) for t in data_dict['text']])
            text = torch.full((text_len.size(0), text_len.max().item()), self.tokenizer.ignore_idx,
                              dtype=text_len.dtype)
            for i in range(text_len.size(0)):
                text[i][:text_len[i]] = data_dict['text'][i]

            data_dict['text'] = text
            data_dict['text_len'] = text_len

            # --- Process the Speaker ID String --- #
            if 'speaker' in data_dict.keys() and hasattr(self, 'spk2idx'):
                assert isinstance(data_dict['speaker'], List)
                # turn the speaker id strings into the trainable tensors
                data_dict['spk_ids'] = torch.LongTensor([self.spk2idx[spk] if spk in self.spk2idx.keys()
                                                         else len(self.spk2idx) for spk in data_dict['speaker']])
                data_dict.pop('speaker')
            return data_dict

        batch_keys = list(batch_data.keys())
        # if the elements are still Dict (multiple dataloaders)
        if isinstance(batch_data[batch_keys[0]], Dict):
            for key in batch_keys:
                batch_data[key] = process_strings(batch_data[key])
        # if the elements are tensors (single dataloader)
        elif isinstance(batch_data[batch_keys[0]], torch.Tensor):
            batch_data = process_strings(batch_data)
        else:
            raise ValueError

        return batch_data


    def model_forward(self,
                      feat: torch.Tensor,
                      text: torch.Tensor,
                      feat_len: torch.Tensor,
                      text_len: torch.Tensor,
                      spk_ids: torch.Tensor = None,
                      epoch: int = None,
                      return_att: bool = False,
                      return_hidden: bool = False,
                      return_enc: bool = False,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """

        Args:
            feat: (batch, feat_maxlen, feat_dim)
                The input speech data. feat_dim = 1 in the case of raw speech waveforms.
            feat_len: (batch,)
                The lengths of input speech data
            text: (batch, text_maxlen)
                The input text data with <sos/eos> at the beginning and end
            text_len: (batch,)
                The lengths of input text data
            spk_ids: (batch,)
                The speaker ids of each speech data. In the form of integer values.
            epoch: int
                The number of the current training epoch.
                Mainly used for mean&std calculation in the feature normalization
            return_att: bool
                Controls whether the attention matrices of each layer in the encoder and decoder will be returned.
            return_hidden: bool
                Controls whether the hidden representations of each layer in the encoder and decoder will be returned.
            return_enc: bool
                Controls whether the final encoder representations will be returned.
            kwargs:
                Temporary register used to store the redundant arguments.

        Returns:
            A dictionary containing all the ASR model outputs necessary to calculate the losses

        """

        # para checking
        assert feat.size(0) == text.size(0) and feat_len.size(0) == text_len.size(0), \
            "The amounts of utterances and sentences are not equal to each other."
        assert feat_len.size(0) == feat.size(0), \
            "The amounts of utterances and their lengths are not equal to each other."
        assert text_len.size(0) == text.size(0), \
            "The amounts of sentences and their lengths are not equal to each other."

        # remove the <sos/eos> at the end of each sentence
        for i in range(text_len.size(0)):
            text[i, text_len[i] - 1] = self.tokenizer.ignore_idx
        text, text_len = text[:, :-1], text_len - 1

        # Encoding
        enc_outputs = self.encoder(feat=feat, feat_len=feat_len,
                                   spk_ids=spk_ids, epoch=epoch)

        # Decoding
        dec_outputs = self.decoder(enc_feat=enc_outputs['enc_feat'],
                                   enc_feat_mask=enc_outputs['enc_feat_mask'],
                                   text=text, text_len=text_len)

        # initialize the asr output to be the decoder predictions
        outputs = dict(
            logits=dec_outputs['output']
        )

        # return the attention results of either encoder or decoder if specified
        if return_att:
            outputs.update(
                att=dict()
            )
            if 'att' in enc_outputs.keys():
                outputs['att'].update(
                    enc_att=enc_outputs['att']
                )
            if 'att' in dec_outputs.keys():
                outputs['att'].update(
                    dec_att=dec_outputs['att']
                )
            assert len(outputs['att']) > 0

        # return the internal hidden results of both encoder and decoder if specified
        if return_hidden:
            outputs.update(
                hidden=dict()
            )
            if 'hidden' in enc_outputs.keys():
                outputs['hidden'].update(
                    enc_hidden=enc_outputs['hidden']
                )
            if 'hidden' in dec_outputs.keys():
                outputs['hidden'].update(
                    dec_hidden=dec_outputs['hidden']
                )
            assert len(outputs['hidden']) > 0

        # return the encoder outputs if specified
        if return_enc:
            assert 'enc_feat' in enc_outputs.keys() and 'enc_feat_mask' in enc_outputs.keys()
            outputs.update(
                enc_feat=enc_outputs['enc_feat'],
                enc_feat_mask=enc_outputs['enc_feat_mask']
            )
        return outputs


    def loss_calculation(self,
                         logits: torch.Tensor,
                         text: torch.Tensor,
                         text_len: torch.Tensor,
                         **kwargs) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        """

        Args:
            logits:
            text:
            text_len:
            **kwargs:

        Returns:

        """
        loss = self.cross_entropy(logits=logits, text=text, text_len=text_len)
        accuracy = self.accuracy(logits=logits, text=text, text_len=text_len)

        # the loss and accuracy must be calculated before being assigned to the returned dict
        # it's better not to use dict(loss=self.cross_entropy(...)) in the dict because it may slow down the program
        losses = dict(loss=loss)
        metrics = dict(loss=loss.clone().detach(), accuracy=accuracy.detach())
        return losses, metrics


    def metrics_calculation(self,
                            logits: torch.Tensor,
                            text: torch.Tensor,
                            text_len: torch.Tensor,
                            **kwargs) -> Dict[str, torch.Tensor]:
        """

        Args:
            logits:
            text:
            text_len:
            **kwargs:

        Returns:

        """
        loss = self.cross_entropy(logits=logits, text=text, text_len=text_len)
        accuracy = self.accuracy(logits=logits, text=text, text_len=text_len)

        return dict(
            loss=loss.detach(),
            accuracy=accuracy.detach()
        )


    def matrix_snapshot(self, vis_logs: List, hypo_attention: Dict, subfolder_names: List[str] or str, epoch: int):
        """
        recursively snapshot all the attention matrices

        Args:
            hypo_attention:
            subfolder_names:

        Returns:

        """
        if isinstance(subfolder_names, str):
            subfolder_names = [subfolder_names]
        keys = list(hypo_attention.keys())

        # process the input data by different data types
        if isinstance(hypo_attention[keys[0]], Dict):
            for key, value in hypo_attention.items():
                self.matrix_snapshot(vis_logs=vis_logs, hypo_attention=value,
                                     subfolder_names=subfolder_names + [key], epoch=epoch)

        # snapshot the information in the materials
        elif isinstance(hypo_attention[keys[0]], np.ndarray):
            vis_logs.append(
                dict(
                    plot_type='matrix', materials=hypo_attention, epoch=epoch,
                    sep_save=False, data_save=False, subfolder_names=subfolder_names
                )
            )


    def attention_reshape(self, hypo_attention: Dict, prefix_list: List = None) -> Dict:
        """

        Args:
            hypo_attention:
            prefix_list:

        """
        if prefix_list is None:
            prefix_list = []

        # process the input data by different data types
        if isinstance(hypo_attention, Dict):
            return {key: self.attention_reshape(value, prefix_list + [key]) for key, value in hypo_attention.items()}
        elif isinstance(hypo_attention, List):
            return {str(index): self.attention_reshape(element, prefix_list + [str(index)])
                    for index, element in enumerate(hypo_attention)}
        elif isinstance(hypo_attention, torch.Tensor):
            hypo_attention = hypo_attention.squeeze()
            if hypo_attention.is_cuda:
                hypo_attention = hypo_attention.detach().cpu()

            if hypo_attention.dim() == 2:
                return {'.'.join(prefix_list + [str(0)]): hypo_attention.numpy()}
            elif hypo_attention.dim() == 3:
                return {'.'.join(prefix_list + [str(index)]): element.numpy()
                        for index, element in enumerate(hypo_attention)}
            else:
                raise RuntimeError


    def visualize(self,
                  epoch: int,
                  sample_index: str,
                  snapshot_interval: int,
                  epoch_records: Dict,
                  feat: torch.Tensor,
                  feat_len: torch.Tensor,
                  text: torch.Tensor,
                  text_len: torch.Tensor,
                  **meta_info):
        """

        Args:
            epoch:
            feat:
            feat_len:
            text:
            text_len:

        Returns:

        """
        # remove the padding zeros at the end of the input feat
        feat = feat[:, :feat_len]

        # obtain the inference results
        infer_results = self.inference(infer_conf=self.visual_infer_conf,
                                       feat=feat, feat_len=feat_len,
                                       text=text, text_len=text_len,
                                       return_att=True)

        # --- snapshot the objective metrics --- #
        vis_logs = []
        # CER, WER, hypothesis probability
        materials = dict()
        for metric in ['cer', 'wer', 'sent_prob', 'len_offset']:
            # store each target metric into materials
            if metric not in epoch_records[sample_index].keys():
                epoch_records[sample_index][metric] = []
            epoch_records[sample_index][metric].append(infer_results[metric]['content'][0])
            materials[metric] = epoch_records[sample_index][metric]
        # save the visualization log
        vis_logs.append(
            dict(
                plot_type='curve', materials=copy.deepcopy(materials), epoch=epoch,
                xlabel='epoch', x_stride=snapshot_interval,
                sep_save=False, subfolder_names=sample_index
            )
        )

        # --- snapshot the subjective metrics --- #
        # record the input audio and real text at the first snapshotting step
        if epoch // snapshot_interval == 1:
            # snapshot input audio
            vis_logs.append(
                dict(
                    plot_type='audio', materials=dict(input_audio=copy.deepcopy(feat[0])),
                    sample_rate=self.sample_rate, audio_format=self.audio_format, subfolder_names=sample_index
                )
            )
            # snapshot real text
            vis_logs.append(
                dict(
                    materials=dict(real_text=[copy.deepcopy(self.tokenizer.tensor2text(text[0][1: -1]))]),
                    plot_type='text', subfolder_names=sample_index
                )
            )
        # hypothesis text
        if 'sent' not in epoch_records[sample_index].keys():
            epoch_records[sample_index]['sent'] = []
        epoch_records[sample_index]['sent'].append(infer_results['sent']['content'][0])
        # snapshot the information in the materials
        vis_logs.append(
            dict(
                materials=dict(hypo_text=copy.deepcopy(epoch_records[sample_index]['sent'])),
                plot_type='text', epoch=epoch, x_stride=snapshot_interval,
                subfolder_names=sample_index
            )
        )

        # hypothesis attention matrix
        infer_results['hypo_att'] = self.attention_reshape(infer_results['hypo_att'])
        self.matrix_snapshot(vis_logs=vis_logs, hypo_attention=copy.deepcopy(infer_results['hypo_att']),
                             subfolder_names=sample_index, epoch=epoch)

        return vis_logs


    def inference(self,
                  infer_conf: Dict,
                  feat: torch.Tensor,
                  feat_len: torch.Tensor,
                  text: torch.Tensor,
                  text_len: torch.Tensor,
                  return_att: bool = False,
                  decode_only: bool = False,
                  teacher_forcing: bool = False,
                  **meta_info) -> Dict[str, Any]:
        """

        Args:
            # --- Testing data arguments --- #
            feat:
            feat_len:
            text:
            text_len:
            meta_info:
            # --- General inference arguments --- #
            return_att:
            decode_only:
            teacher_forcing:
            # --- Beam searching arguments --- #
            infer_conf:

        Returns:

        """
        # go through beam searching process
        if not teacher_forcing:
            # copy the input data in advance for data safety
            model_input = copy.deepcopy(
                dict(feat=feat, feat_len=feat_len)
            )

            # Encoding input speech
            enc_outputs = self.encoder(**model_input)

            # generate the model hypothesis
            infer_results = beam_searching(enc_feat=enc_outputs['enc_feat'],
                                           enc_feat_mask=enc_outputs['enc_feat_mask'],
                                           decode_one_step=self.decoder,
                                           vocab_size=self.tokenizer.vocab_size,
                                           sos_eos=self.tokenizer.sos_eos_idx,
                                           padding_idx=self.tokenizer.ignore_idx,
                                           **infer_conf)
            hypo_text = infer_results['hypo_text']
            hypo_text_len = infer_results['hypo_text_len']
            hypo_len_ratio = infer_results['hypo_len_ratio']
            hypo_text_prob = infer_results['hypo_text_prob']

        # calculate the attention matrix
        if teacher_forcing or return_att:
            infer_results = self.model_forward(feat=feat, feat_len=feat_len,
                                               text=text if teacher_forcing else hypo_text,
                                               text_len=text_len if teacher_forcing else hypo_text_len,
                                               return_att=return_att)
            # return the attention matrices
            if return_att:
                hypo_att = infer_results['att']

            # update the hypothesis text-related data in the teacher forcing mode
            if teacher_forcing:
                # the last token is meant to be eos which should not appear in the hypothesis text
                infer_results['logits'] = torch.log_softmax(infer_results['logits'][:, :-1], dim=-1)
                hypo_text_prob, hypo_text = torch.max(infer_results['logits'], dim=-1)
                # the original text contains both sos at the beginning and eos at the end
                hypo_text_len = text_len - 2
                hypo_len_ratio = torch.ones_like(hypo_text_len)
                hypo_text_prob = torch.sum(hypo_text_prob, dim=-1) / (hypo_text_len ** length_penalty)

        # check the data
        assert hypo_text.size(0) == text.size(0), \
            f"The first dimension of text and hypo_text doesn't match! " \
            f"Got text.size(0)={text.size(0)} and hypo_text.size(0)={hypo_text.size(0)}."

        # obtain the cer and wer metrics
        cer_wer = self.error_rate(hypo_text=hypo_text, real_text=text, tokenizer=self.tokenizer)

        # recover the text tensors back to text strings (removing the padding and sos/eos tokens)
        hypo_text = [self.tokenizer.tensor2text(hypo[torch.logical_and(hypo != self.tokenizer.ignore_idx,
                                                                       hypo != self.tokenizer.sos_eos_idx)])
                     for hypo in hypo_text]
        text = [self.tokenizer.tensor2text(real[torch.logical_and(real != self.tokenizer.ignore_idx,
                                                                  real != self.tokenizer.sos_eos_idx)])
                for real in text]
        text_len -= 2

        # in the decoding-only mode, only the hypothesis-related results will be returned
        outputs = dict(
            sent=dict(format='txt', content=hypo_text),
            sent_len=dict(format='txt', content=to_cpu(hypo_text_len + 1)), # consider one <sos/eos> at the end
            len_ratio=dict(format='txt', content=to_cpu(hypo_len_ratio)),
            sent_prob=dict(format='txt', content=to_cpu(hypo_text_prob))
        )
        if decode_only:
            return outputs

        # in the normal mode, return all the information calculated by the reference
        outputs.update(
            cer=dict(format='txt', content=cer_wer['cer']),
            wer=dict(format='txt', content=cer_wer['wer'])
        )
        # add the attention matrix into the output Dict, only used for model visualization during training
        # because it will consume too much time for saving the attention matrices of all testing samples during testing
        if return_att:
            outputs.update(
                hypo_att=hypo_att
            )

        # evaluation reports for all the testing samples
        sample_reports, insertion, deletion, substitution = [], [], [], []
        for i in range(len(text)):
            i_num, d_num, s_num, align_table = \
                get_word_edit_alignment(hypo_text[i], text[i])
            md_list = get_list_strings(
                {
                    'CER': f"{cer_wer['cer'][i]:.2%}",
                    'WER': f"{cer_wer['wer'][i]:.2%}",
                    'Hypothesis Probability': f"{to_cpu(hypo_text_prob)[i]:.6f}",
                    'Length Offset': f"{'+' if to_cpu(hypo_text_len - text_len)[i] >= 0 else ''}"
                                     f"{to_cpu(hypo_text_len - text_len)[i]:d}",
                    'Length Ratio': f"{to_cpu(hypo_len_ratio)[i]:.2f}",
                    'Word Insertion': f"{i_num}",
                    'Word Deletion': f"{d_num}",
                    'Word Substitution': f"{s_num}"
                }
            )

            sample_reports.append(
                '\n\n' +
                md_list +
                '\n' +
                align_table +
                '\n'
            )
            insertion.append(i_num)
            deletion.append(d_num)
            substitution.append(s_num)

        outputs['sample_reports.md'] = dict(format='txt', content=sample_reports)
        outputs.update(
            insertion=dict(format='txt', content=insertion),
            deletion=dict(format='txt', content=deletion),
            substitution=dict(format='txt', content=substitution)
        )
        return outputs


class SemiASR(ASR):
    """

    """
    def model_forward(self, **batch_data) -> Dict[str, Dict or torch.Tensor]:
        """

        Args:
            **batch_data:

        Returns:

        """
        # if the sub-batches are not Dict, go through the supervised training process
        batch_keys = list(batch_data.keys())
        if not isinstance(batch_data[batch_keys[0]], Dict):
            return super().model_forward(**batch_data)

        # otherwise, go through the semi-supervised training process
        outputs = dict()
        for name, sub_batch in batch_data.items():
            outputs[name] = super().model_forward(**sub_batch)

        return outputs

    def loss_calculation(self,
                         batch_data: Dict[str, Dict[str, torch.Tensor]],
                         model_outputs: Dict[str, Dict[str, torch.Tensor]],
                         **kwargs) -> Dict[str, torch.Tensor]:
        """

        Args:
            batch_data:
            model_outputs:
            **kwargs:

        Returns:

        """
        # supervised criteria calculation
        sup_criterion = self.cross_entropy['sup']['criterion']
        sup_loss = sup_criterion(pred=model_outputs['sup']['logits'],
                                 text=batch_data['sup']['text'],
                                 text_len=batch_data['sup']['text_len'])
        sup_accuracy = self.accuracy(pred=model_outputs['sup']['logits'],
                                     text=batch_data['sup']['text'],
                                     text_len=batch_data['sup']['text_len'])

        # unsupervised criteria calculation
        unsup_criterion = self.cross_entropy['unsup']['criterion']
        unsup_loss = unsup_criterion(pred=model_outputs['unsup']['logits'],
                                     text=batch_data['unsup']['text'],
                                     text_len=batch_data['unsup']['text_len'])
        unsup_accuracy = self.accuracy(pred=model_outputs['unsup']['logits'],
                                       text=batch_data['unsup']['text'],
                                       text_len=batch_data['unsup']['text_len'])

        # total loss calculation
        total_loss = sup_loss * self.cross_entropy['sup']['weight'] + \
                     unsup_loss * self.cross_entropy['unsup']['weight']

        return dict(
            total_loss=total_loss,
            sup_loss=sup_loss,
            unsup_loss=unsup_loss,
            sup_accuracy=sup_accuracy,
            unsup_accuracy=unsup_accuracy
        )
