import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
import random
from tqdm import tqdm
import os
import copy
import torch.nn.functional as F

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

class SFTData(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        # self.K = K
        self.dedup = dedup
        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()  
    def __len__(self):
        return len(self.data)


    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""
    
    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""



    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        history_str = "::".join(row["history_item_title"])
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\"\n"
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item,
                "history_str": history_str,
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
""" 
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
                
           
        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(len(tokens))
        
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            
        }
    

    
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            # print(inputs[-1])
            
        self.inputs = inputs
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]


class D3Dataset(Dataset):
    def __init__(self, train_file, max_len=2048, sample=-1, seed=0, category="", dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.category = category
        self.dedup = dedup
        self.prompt2history = {}
        self.history2target = {}
        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()  
        

    def __len__(self):
        return len(self.data)
    
    
        
        
    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""


    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        history_str = "::".join(row["history_item_title"])
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\"\n"
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item,
                "history_str": history_str,
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
"""        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
           
        prompt = self.generate_prompt(history)
        self.prompt2history[instruction + prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item
        
        return {
            "prompt": instruction + prompt,
            "completion": target_item,
        }
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            
        self.inputs = inputs
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]

class DPODataset(Dataset):
    def __init__(self, train_file, info_file, tokenizer, neg_num=3, max_len=2048, sample=-1, test = False, seed=0, category="", dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        with open(info_file, 'r') as f:
            info = f.readlines()
            # Split on the LAST tab, not the first: a handful of titles contain a
            # literal tab (2 lines in Toys_and_Games), and splitting on the first
            # one truncates them so they no longer match `item_title` in the csv.
            # This is the same parse evaluate.py uses to build the decoding pool.
            info = ["\"" + _[:-len(_.split('\t')[-1])].strip() + "\"\n" for _ in info]
            self.item_name = info
        
        self.neg_num = neg_num
        self.max_len = max_len
        self.category = category
        # self.K = K
        self.dedup = dedup
        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()  
    def __len__(self):
        return len(self.data)


    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response: 
{data_point["output"]}
"""
    
    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""


    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""  
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\""
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
"""
        # tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx]) # 从交互数据中拆分出历史交互数据和目标物品
        target_item = history['output']
        history['output'] = ''
        # negative_prompt_ids = copy.deepcopy(tokens)
        negative_items = [item for item in self.item_name if item != target_item]
        neg_sam = random.sample(negative_items, self.neg_num)
        
        dic = {}
        prompt = self.generate_prompt(history)
        # prompt 里不包含任何输出，只有输入数据
        dic["prompt"] = instruction + prompt
        # keep the two pieces so the collator can encode them separately and
        # concatenate at the id level, exactly like SFTData/EvalD3Dataset do
        dic["instruction"] = instruction
        dic["user_prompt"] = prompt
        dic["chosen"] = target_item
        for i in range(self.neg_num):
            dic[f"rejected{i}"] = neg_sam[i]

        return dic
        # tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        # history["input"] = ""
        
        # attention_mask = [1] * len(tokens)
        
        
        # if self.test:
        #     return {
        #         "input_ids": tokens,
        #         "attention_mask": attention_mask,
                
        #         # "select_index": select_index,
        #     }    
        
        # golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        # input_prompt_len = len(tokens)
        # tokens = tokens + golden_tokens
        # attention_mask = [1] * len(tokens)
        # labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        # if len(tokens) >= self.max_len:
        #     print(len(tokens))
        
        
        # return {
        #     "input_ids": tokens[-self.max_len:],
        #     "attention_mask": attention_mask[-self.max_len:],
        #     "labels": labels[-self.max_len:],
            
        # }
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            # print(inputs[-1])
            
        self.inputs = inputs
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]

class SPRecDataset(Dataset):
    """DPO preference data whose negatives are the SFT model's own top predictions.

    Reads the sampled output of a trained SFT model (the json written by
    evaluate.py) and turns each record into one positive plus `neg_num` hard
    negatives. Unlike DPODataset, which draws negatives uniformly from the whole
    item pool, these negatives are what the model actually ranks highest, so the
    preference pairs carry a much stronger learning signal.

    The json holds `input` (interaction history), `output` (target item) and
    `predict` (items sampled from the SFT model, ordered by descending
    probability). The negatives are the first `neg_num` entries of
    `predict[:top_k]` once the target has been removed -- the target only shows
    up when the SFT model ranked it inside its top-k, which happens for ~6% of
    records.
    """

    def __init__(self, result_file, tokenizer=None, neg_num=3, top_k=4, max_len=2048,
                 sample=-1, seed=0, category="", dedup=False, **kwargs):
        random.seed(seed)
        with open(result_file, 'r') as f:
            self.data = json.load(f)

        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)

        self.tokenizer = Tokenizer(tokenizer) if tokenizer is not None else None
        self.neg_num = neg_num
        self.top_k = top_k
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def normalize(item):
        """Strip the quoting and newline so `output` and `predict` are comparable.

        `output` is stored as '"Some Item"\\n' while `predict` entries carry no
        trailing newline, so comparing them raw never matches -- which would
        silently leave the target sitting among the negatives.
        """
        return str(item).strip().strip('"').strip()

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def pick_negatives(self, record):
        """Take the top-ranked predictions, dropping the target if it is there."""
        target = self.normalize(record["output"])
        negatives = [item for item in record["predict"][:self.top_k]
                     if self.normalize(item) != target]
        return negatives[:self.neg_num]

    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
"""
        record = self.data[idx]
        neg_sam = self.pick_negatives(record)
        if len(neg_sam) < self.neg_num:
            return None

        target_item = record["output"]
        if not target_item.endswith('\n'):
            target_item = target_item + '\n'

        dic = {}
        prompt = self.generate_prompt({"input": record["input"], "output": ""})
        dic["prompt"] = instruction + prompt
        # keep the two pieces so the collator can encode them separately and
        # concatenate at the id level, exactly like SFTData/EvalD3Dataset do
        dic["instruction"] = instruction
        dic["user_prompt"] = prompt
        dic["chosen"] = target_item
        for i in range(self.neg_num):
            # match the trailing newline of `chosen` so the DPO log-prob
            # comparison is not skewed by an extra token
            dic[f"rejected{i}"] = neg_sam[i].rstrip('\n') + '\n'

        return dic

    def get_inputs(self):
        inputs = []
        skipped = 0
        for i in tqdm(range(len(self.data))):
            dic = self.pre(i)
            if dic is None:
                skipped += 1
                continue
            inputs.append(dic)

        if skipped:
            print(f"SPRecDataset: skipped {skipped} records with fewer than "
                  f"{self.neg_num} negatives among the top-{self.top_k}")
        self.inputs = inputs

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]

class EvalD3Dataset(Dataset):

    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()  
    def __len__(self):
        return len(self.data)


    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""
    
    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""


    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\""
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
                
           
        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                
                # "select_index": select_index,
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(len(tokens))
        
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            
        }
    

    
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            
        self.inputs = inputs
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]