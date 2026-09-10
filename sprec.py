import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig, BitsAndBytesConfig
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from datasets import load_dataset, concatenate_datasets
from torch.optim.lr_scheduler import LambdaLR
from dpotrainer.softmax_dpo_trainer import SDPOTrainer
from accelerate import Accelerator


"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import D3Dataset, SPRecDataset, DPODataset

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



def train(
    # model/data params
    base_model: str = "/storage_fast/yxchen/huggingface_data/hub/Qwen2-0.5B",  # the only required argument
    train_file: str="/storage_fast/yxchen/DecodingMatters/data/Amazon/train/Toys_and_Games_5_2016-10-2018-11.csv",
    eval_file: str="/storage_fast/yxchen/DecodingMatters/data/Amazon/valid/Toys_and_Games_5_2016-10-2018-11.csv",
    info_file: str = "/storage_fast/yxchen/DecodingMatters/data/Amazon/info/Toys_and_Games_5_2016-10-2018-11.txt",
    result_file: str = None,
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    beta: float = 1.0,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    neg_num: int = 1,
    eval_step: float=0.05,
    save_step: float=0.05,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    
    local_rank: int = 0,
    deepspeed: str ="./deepspeed.json",
    category: str="Toys_and_Games",
    K: int = 0,
    version: str = "base",
    train_from_scratch: bool = False,

):
    os.environ['WANDB_PROJECT'] = wandb_project
    # print(train_file)
    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", \
        "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", \
        "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", \
        "Movies": "movie", "Yelp": "restaurants", "Industrial_and_Scientific": "industrial and scientific items", "Clothing_Shoes_and_Jewelry": "clothes and shoes"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        # print("ddp")
        # print(world_size)
        # print()
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    # ddp = None
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    
    # uses.environ["WANDB_LOG_MODEL"] = wandb_log_model
    # os.environ["WANDB_DISABLED"] = "true"
    # device_index = Accelerator().process_index
    # device_map = {"": device_index}

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            # quantization_config=bnb_config
        )
        reference_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            # quantization_config=bnb_config
        )
        reference_model.eval()
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        reference_model = AutoModelForCausalLM.from_config(config)
        reference_model.eval()
        print("Training from scratch!")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    train_data = SPRecDataset(train_file=train_file, info_file=info_file, tokenizer=tokenizer, \
        result_file=result_file, neg_num=neg_num, max_len=cutoff_len, seed=seed, category=category)
    val_data = DPODataset(train_file=eval_file, info_file=info_file, tokenizer=tokenizer, neg_num=neg_num, max_len=cutoff_len, seed=seed, category=category)
        
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        print(1111)
        model.is_parallelizable = True
        model.model_parallel = True
    
    from datasets import Dataset as HFDataset
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    print(hf_train_dataset)
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})
    trainer = SDPOTrainer(
        # deepspeed=deepspeed,
        model,
        reference_model,
        tokenizer=tokenizer,
        beta=beta,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len*2,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            # per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing =True,
            max_grad_norm= 0.3,
            num_train_epochs=num_epochs, 
            learning_rate=learning_rate,
            bf16=True,
            save_strategy="steps",
            save_steps=save_step,
            save_total_limit=20,
            eval_strategy="steps",
            eval_steps=eval_step,
            max_steps=100,
            # load_best_model_at_end=True,
            logging_steps=1,
            output_dir=output_dir,
            report_to = "wandb",
            run_name = wandb_run_name,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            remove_unused_columns=False,
            gradient_checkpointing_kwargs={'use_reentrant': True}, 
            ddp_find_unused_parameters=False,
        ),
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=1)],
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)