
import torch
import logging
import os
import copy
import json
from .utils import DataProcessor
from typing import *


logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_jsno_string())


    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.jumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        
    def __repr__(self):
        return str(self.to_json_string())
        
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def xlnet_collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, -max_len:]
    all_attention_mask = all_attention_mask[:, -max_len:]
    all_token_type_ids = all_token_type_ids[:, -max_len:]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def truncate_func(text1:List, text2:List, max_seq_length: int) -> List:
    # 1. 两句都超了
    # 2. 其中一句超了
    while len(text1) + len(text2) > max_seq_length:
        if len(text1) >= len(text2):
            text1 = text1[:-1]
        else:
            text2 = text2[:-1]

    return text1, text2

def convert_examples_to_feature(examples, max_seq_length, tokenizer,
                                cls_token_at_end=False, cls_token="[CLS]",
                                cls_token_segment_id=1,sep_token="[SEP]",
                                pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                sequence_a_segment_id=0,sequence_b_segment_id = 1, 
                                mask_padding_with_zero=True,):
    features = []
    for (ex_index, example) in enumerate(examples):
        texts_a = example.text_a
        texts_b = example.text_b
        tokens_corpus = tokenizer.tokenize(texts_a)
        tokens_entity = tokenizer.tokenize(texts_b)

        special_tokens_count =3 
        if len(tokens_corpus+tokens_entity) > max_seq_length - special_tokens_count:
            tokens_corpus, tokens_entity = truncate_func(tokens_corpus, tokens_entity, max_seq_length - special_tokens_count)
        # if len(tokens_corpus) > max_seq_length - special_tokens_count:
        #     tokens_corpus = tokens_corpus[:(max_seq_length - special_tokens_count)]
        # if len(tokens_entity) > max_seq_length - special_tokens_count:
        #     tokens_entity = tokens_entity[:(max_seq_length - special_tokens_count)]    

        label_id = example.label
        tokens =  tokens_corpus + [sep_token] 

        segment_ids =  [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids


        tokens += tokens_entity + [sep_token]  # input_id [cls]+ text_a + [sep]+ text_b + [sep]
                                               # segment  000           +  0   + 1111   +  1
        segment_ids += [sequence_b_segment_id] * len(tokens_entity+[sep_token])

        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = [pad_token] * padding_length + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_ids
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        features.append(InputFeatures(input_ids=input_ids, 
                                     attention_mask = input_mask, 
                                     token_type_ids=segment_ids,
                                     label=label_id))

    return features

        
class SentenceBertProcessor(DataProcessor):
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['corpus']
            text_b = line['entity']
            label = line['label']
            # subject = get_entities(labels,id2label=None,markup='bios')
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


prompt_processors = {
    "bert_prompt": SentenceBertProcessor
}

