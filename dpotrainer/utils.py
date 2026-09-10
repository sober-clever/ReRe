import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase, TrainerCallback

@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt + chosen/rejected responses.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """Encode one piece of text exactly like `data.Tokenizer.encode`.

        The SFT/eval path (SFTData.pre, EvalD3Dataset.pre) builds its token
        sequence with this wrapper, so DPO has to use the same one or the two
        paths can drift by a token: the wrapper strips any tokenizer-inserted
        BOS/EOS and then re-adds them explicitly.
        """
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        while ids and bos_id is not None and ids[0] == bos_id:
            ids = ids[1:]
        while ids and eos_id is not None and ids[-1] == eos_id:
            ids = ids[:-1]

        if bos and bos_id is not None:
            ids = [bos_id] + ids
        if eos and eos_id is not None:
            ids = ids + [eos_id]
        return ids

    def _as_tokens(self, ids: List[int]) -> Dict[str, List[int]]:
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: Dict[str, str],
        instruction: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.

        When `instruction` and `user_prompt` are given, they are encoded separately
        and concatenated at the id level -- the same two-step build the SFT/eval
        path uses -- instead of encoding the joined string in one call.
        """
        if instruction is not None and user_prompt is not None:
            prompt_ids = self.encode(instruction, bos=True, eos=False) + self.encode(
                user_prompt, bos=False, eos=False
            )
        else:
            prompt_ids = self.encode(prompt, bos=True, eos=False)
        assert self.tokenizer.eos_token_id not in prompt_ids, f"Prompt contains EOS token: {prompt}"
        prompt_tokens = self._as_tokens(prompt_ids)

        # eos is appended here (eos=True) rather than by hand further down
        chosen_tokens = self._as_tokens(self.encode(chosen, bos=False, eos=True))
        rejected_tokens = {}
        for key in rejected:
            rejected_tokens[key] = self._as_tokens(self.encode(rejected[key], bos=False, eos=True))

        # exactly one trailing EOS, the one encode() just added
        assert (
            chosen_tokens["input_ids"].count(self.tokenizer.eos_token_id) == 1
        ), f"Chosen response contains EOS token: {chosen}"
        assert all(
            rejected_tokens[key]["input_ids"].count(self.tokenizer.eos_token_id) == 1
            for key in rejected_tokens
        ), f"Rejected response contains EOS token: {rejected}"

        max_rejected_len = max([len(rejected_tokens[key]["input_ids"]) for key in rejected_tokens])
        longer_response_length = max(len(chosen_tokens["input_ids"]), max_rejected_len)

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()}

        # Create labels
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {}
        # rejected_tokens: Dict[str, Dict]
        for key in rejected_tokens:
            rejected_sequence_tokens[key] = {k: prompt_tokens[k] + rejected_tokens[key][k] for k in rejected_tokens[key]}
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        for key in rejected_sequence_tokens:
            rejected_sequence_tokens[key]["labels"] = rejected_sequence_tokens[key]["input_ids"][:]
            rejected_sequence_tokens[key]["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )

        batch = {}

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        for key in rejected:
            batch[key] = prompt + rejected[key]
        batch["chosen_response_only"] = chosen
        for key in rejected:
            batch[f"{key}_response_only"] = rejected[key]

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            # "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        # rejected_sequence_tokens: Dict[str, Dict]
        for k, toks in rejected_sequence_tokens.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        
        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # adapted from https://stackoverflow.com/questions/73256206
                if "prompt" in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = self.padding_value
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                if "prompt" in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("sim"):
                # print(k)
                padded_batch[k] = torch.Tensor([ex[k] for ex in batch])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = {}
            for key in feature:
                if key.startswith("rejected") and not key.endswith("sim"):
                    rejected[key] = feature[key]

            batch_element = self.tokenize_batch_element(
                prompt,
                chosen,
                rejected,
                instruction=feature.get("instruction"),
                user_prompt=feature.get("user_prompt"),
            )
            
            for key in feature:
                if key.endswith("sim"):
                    batch_element[key] = feature[key]
            
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)
    
def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )