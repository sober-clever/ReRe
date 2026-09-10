from typing import List, Dict, Any, Callable

import torch
from loguru import logger
from transformers import PreTrainedTokenizerFast

from attention_masks_seq2seq import get_cross_attention_mask, get_isolated_self_attention_mask, get_last_token_only_cross_attention_mask
from .utils import get_sentence_start_positions, process_metas

meta_cutoff_len = None


def get_position_ids(cut_indexes: List[List[int]],  # [cut indexes] for each batch
                     padding_side: str,
                     padding_value: int = 0):
    all_position_ids = []

    for batch_i, batch_cut_indexes in enumerate(cut_indexes):
        num_items = len(batch_cut_indexes) - 1
        positions = []
        meta_lens = [batch_cut_indexes[i + 1] - batch_cut_indexes[i] for i in range(num_items)]
        for i in range(num_items):
            meta_len = meta_lens[i]
            meta_pos1 = list(range(0, meta_len))
            positions += meta_pos1
        all_position_ids.append(positions)

    max_len = max([len(positions) for positions in all_position_ids])
    for batch_i in range(len(all_position_ids)):
        paddings = [padding_value] * (max_len - len(all_position_ids[batch_i]))
        if padding_side == 'left':
            all_position_ids[batch_i] = paddings + all_position_ids[batch_i]
        elif padding_side == 'right':
            all_position_ids[batch_i] = all_position_ids[batch_i] + paddings

    all_position_ids = torch.tensor(all_position_ids, dtype=torch.long)  # shape: (B, seq_len)

    assert all_position_ids.shape[1] == max_len == max(cut_index[-1] for cut_index in cut_indexes)
    return all_position_ids


class CrossAttnDataCollatorAlwaysBOS:
    get_attention_mask = staticmethod(get_last_token_only_cross_attention_mask)
    get_encoder_attention_mask = staticmethod(get_isolated_self_attention_mask)

    def __init__(self, itemid2meta_str: Callable[[int], str], tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

        logger.info(f"Using {self.__class__.__name__}")

        logger.info(f'cross attention mask generation method: {self.get_attention_mask.__name__}')
        logger.info(self.get_attention_mask.__doc__)

        logger.info(f'encoder self attention mask generation method: {self.get_encoder_attention_mask.__name__}')
        logger.info(self.get_encoder_attention_mask.__doc__)

        self.itemid2meta_str = itemid2meta_str
        self.item_meta: Dict[int, str] = {
            -1: 'A user shopping history:',
            0: 'The user registered an account at Amazon, and started shopping.'
        }

    def get_meta_str(self, i: int):
        if i not in self.item_meta:
            self.item_meta[i] = self.itemid2meta_str(i)
        return self.item_meta[i]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        meta_ids_strs = []
        item_labels: List[int] = []
        item_ids: List[int] = []
        tokenizer = self.tokenizer
        for feature in features:
            feature_meta_: List[str] = [self.get_meta_str(i) for i in feature["input_ids"]]
            feature_meta_ = [self.get_meta_str(-1)] + feature_meta_

            _all_meta_ids = process_metas(feature_meta_, tokenizer.eos_token, meta_cutoff_len)
            meta_ids_str = ''.join(_all_meta_ids)
            meta_ids_strs.append(meta_ids_str)

            item_ids.append([0] + feature["input_ids"])
            item_labels.append([-100] + feature["labels"])
        pad_token = tokenizer.pad_token_id
        encoded_metas = tokenizer(meta_ids_strs,
                                  padding=True,
                                  return_tensors="pt",
                                  return_attention_mask=True,
                                  add_special_tokens=False,
                                  )
        tokenizer.pad_token_id = 0
        encoded_items = tokenizer.pad({"input_ids": item_ids}, return_tensors="pt", padding=True,
                                      return_attention_mask=True)
        item_labels = tokenizer.pad({"input_ids": item_labels}, return_tensors="pt", padding=True)
        tokenizer.pad_token_id = pad_token
        encoder_input_ids, encoder_attention_mask = encoded_metas["input_ids"], encoded_metas["attention_mask"]
        decoder_input_ids, decoder_attention_mask = encoded_items["input_ids"], encoded_items["attention_mask"]

        item_labels = item_labels["input_ids"]
        item_labels[item_labels <= 0] = -100

        # encoder_position_ids = torch.arange(encoder_input_ids.shape[1], device=encoder_input_ids.device).unsqueeze(0)
        # if encoder_input_ids.shape[1] > 1024:
        #     logger.warning(f"encoder_input_ids.shape[1] > 1024: {encoder_input_ids.shape[1]}")
        #     encoder_position_ids[:, 1024:] = 1023



        sentences_start_indices = get_sentence_start_positions(input_ids=encoder_input_ids, tokenizer=tokenizer)
        assert all([len(idxs) == len(items) + 1 for idxs, items in zip(sentences_start_indices, item_ids)])

        cross_attn_mask = self.get_attention_mask(meta_attention_mask=encoder_attention_mask,
                                                  item_attention_mask=decoder_attention_mask,
                                                  cut_indexes=sentences_start_indices,
                                                  dtype=torch.float32,
                                                  padding_side=tokenizer.padding_side,
                                                  query_first=False)
        encoder_position_ids = get_position_ids(cut_indexes=sentences_start_indices,
                                                padding_side=tokenizer.padding_side,
                                                padding_value=0)
        encoder_attention_mask = self.get_encoder_attention_mask(
            attention_mask=encoder_attention_mask,
            cut_indexes=sentences_start_indices,
            dtype=torch.float32,
        )

        # assert shapes
        assert decoder_input_ids.shape[1] == item_labels.shape[1]  # (B, S)[1] == (B, S)[1]

        assert [len(idxs) == len(items) + 1 for idxs, items in zip(sentences_start_indices, item_ids)]

        """
        decoder_position_ids 暂且不管 (if right pad
        """
        return {
            "input_ids": {
                "encoder_input_ids": encoder_input_ids,
                "decoder_input_ids": decoder_input_ids,
            },
            "labels": {
                "decoder_labels": item_labels,
            },
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_position_ids": encoder_position_ids,
            "decoder_self_attention_mask": decoder_attention_mask,
            "decoder_cross_attention_mask": cross_attn_mask,  # (B, 1, target_len, query_len)

        }