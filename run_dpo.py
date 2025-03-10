#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import sys
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from configs import DataArguments, DPOConfig, H4ArgumentParser, ModelArguments, SFTConfig
from data import apply_chat_template, get_datasets
from model_utils import get_kbit_device_map, is_adapter_model, DEFAULT_CHAT_TEMPLATE
from peft import LoraConfig, PeftConfig, PeftModel,prepare_model_for_kbit_training
from trl import DPOTrainer
import datasets
from itertools import combinations
from collections import defaultdict
import numpy as np
import random
import re
import json

IGNORE_TOKEN_ID = -100
preprocessing_num_workers = 1
logger = logging.getLogger(__name__)

def concatenated_forward(self, model, batch):

    concatenated_batch = self.concatenated_inputs(
        batch,
        is_encoder_decoder=self.is_encoder_decoder,
        label_pad_token_id=self.label_pad_token_id,
        padding_value=self.padding_value,
        device=self.accelerator.device,
    )
    len_chosen = batch["chosen_labels"].shape[0]

    if os.getenv('DRY_RUN'):
        print('----------------------------------------------------------------------')
        print(self.tokenizer.batch_decode(concatenated_batch['concatenated_input_ids']))
        print('----------------------------------------------------------------------')
        for l in concatenated_batch['concatenated_labels']:
            idxs = torch.where(l!=-100)[0]
            print(self.tokenizer.decode(l[idxs]))
        print('----------------------------------------------------------------------')
        # concatenated_batch['concatenated_labels'] * concatenated_batch['concatenated_attention_mask']
        # concatenated_batch['concatenated_input_ids'] * concatenated_batch['concatenated_attention_mask']
    
    model_kwargs = (
        {
            "labels": concatenated_batch["concatenated_labels"],
            "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
        }
        if self.is_encoder_decoder
        else {}
    )
    all_logits = model(
        concatenated_batch["concatenated_input_ids"],
        attention_mask=concatenated_batch["concatenated_attention_mask"],
        **model_kwargs,
    ).logits

    all_logps = self.get_batch_logps(
        all_logits,
        concatenated_batch["concatenated_labels"],
        average_log_prob=False,
        is_encoder_decoder=self.is_encoder_decoder,
        label_pad_token_id=self.label_pad_token_id,
    )

    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]

    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

def process_ultrachat(example):
    for data in [ example['chosen'], example['rejected'] ]:
        for d in data:
            d['value'] = d['content']
            del d['content']
            d['from'] = d['role']
            del d['role']
    return example

def process_intel_orca(example):
    return {
        'prompt': example['question'],
        'chosen': [
            {
                "from": "system",
                "value": example['system'],
            },
            {
                "from": "user",
                "value": example['question'],
            },
            {
                "from": "assistant",
                "value": example['chosen'],
            },
        ],
        'rejected': [
            {
                "from": "system",
                "value": example['system'],
            },
            {
                "from": "user",
                "value": example['question'],
            },
            {
                "from": "assistant",
                "value": example['rejected'],
            },
        ],
    }

def contains_chatgpt_words(text):
    replace_keywords = {
        'ChatGPT': 'CCIIP-GPT',
        'OpenAI': '华中科技大学CCIIP-LAB',
        'GPT3.5': 'CCIIP-GPT',
        'GPT4':'CCIIP-GPT',
    }
    for word in replace_keywords:
        if word.lower() in text.lower():
            return True
    return False

def process_berkeley_nectar(example):
    # input 1 return 
    converted_sample = defaultdict(list)
    # because batched=True
    turns = example['turns'][0]
    prompt = example["prompt"][0]
    # may has many turns; 
    answers = example["answers"][0]
    # adopted different format, you can either input history in one turn by: \n\nHuman(User): \n\nAssistant ; or chat multiple turn
    history = []
    rand = random.random()
    # if rand < 0.1:
    #     prompt = prompt.replace('Human', 'User')
    # elif rand > 0.2:
    if turns == 1:
        # 里边有很多不属于分隔符的Human Assistant 混杂, 无法直接re.split
        prompt = re.sub('^\n\nHuman: +', '', prompt)
        prompt = re.sub('\n\nAssistant: +$', '', prompt)
    else:
        conversation = re.split(r'\n\nHuman: +|\n\nAssistant: +',prompt)
        conversation = [d for d in conversation if d != '']
        if len(conversation) % 2 :
            prompt = prompt[-1]
            user = conversation[1:-1:2]
            bot = conversation[2::2]
            for u,b in zip(user,bot):
                history.append({
                    "from": "user",
                    "value": u,
                })
                history.append({
                    "from": "assistant",
                    "value": b,
                })
        else:
            pass
    history.append({
        "from": "user",
        "value": prompt,
    })
    # already sorted (too much C_7^2=21)
    # for comp_idx1, comp_idx2 in combinations(range(len(answers)), 2):
    #     ans1, ans2 = answers[comp_idx1], answers[comp_idx2]
    #     converted_sample["prompt"].append(prompt)
    #     converted_sample["chosen"].append(ans1['answer'])
    #     converted_sample["rejected"].append(ans2['answer'])
    for idx in range(2):
        for comp_idx2 in range(3, len(answers)):
            ans1, ans2 = answers[idx], answers[comp_idx2]
            converted_sample["prompt"].append(prompt)
            converted_sample["chosen"].append(history + [
                {
                    "from": "assistant",
                    "value": ans1['answer'],
                }
            ])
            converted_sample["rejected"].append(history+[
                {
                    "from": "assistant",
                    "value": ans2['answer'],
                }
            ])
    return converted_sample

def process_argilla_math(example):
    return {
        'prompt': example['instruction'],
        'chosen': [
            {
                "from": "user",
                "value": example['instruction'],
            },
            {
                "from": "assistant",
                "value": example['chosen_response'],
            },
        ],
        'rejected': [
            {
                "from": "user",
                "value": example['instruction'],
            },
            {
                "from": "assistant",
                "value": example['rejected_response'],
            },
        ],
    }

def get_datasets(
    data_config,
) :
    ans = datasets.DatasetDict()
    for mixer, mode in zip(
        [data_config.train_dataset_mixer, data_config.eval_dataset_mixer],
        ['train', 'test'],
    ):
        raw_datasets = []
        for ds in mixer:
            dataset = datasets.load_dataset(ds['path'], split=ds['split'])
            dataset = dataset.select(range(int(ds['frac'] * len(dataset))))
            if 'ultra' in ds['path']:
                dataset = dataset.map(
                    process_ultrachat,
                    remove_columns=set(dataset.column_names) - set(["prompt", "chosen", "rejected"]),
                    num_proc=preprocessing_num_workers,
                )
                print(dataset)
            elif 'orca' in ds['path']:
                dataset = dataset.map(
                    process_intel_orca,
                    remove_columns=set(dataset.column_names) - set(["prompt", "chosen", "rejected"]),
                    num_proc=preprocessing_num_workers,
                )
                print(dataset)
            elif 'Nectar' in ds['path']:
                dataset = dataset.map(
                    process_berkeley_nectar,
                    batched=True,
                    batch_size=1,
                    remove_columns=set(dataset.column_names) - set(["prompt", "chosen", "rejected"]),
                    num_proc=preprocessing_num_workers,
                )
                print(dataset)
            elif 'distilabel-math-preference-dpo' in ds['path']:
                dataset = dataset.map(
                    process_argilla_math,
                    remove_columns=set(dataset.column_names) - set(["prompt", "chosen", "rejected"]),
                    num_proc=preprocessing_num_workers,
                )
                print(dataset)   
            else:
                raise NotImplementedError
            raw_datasets.append(dataset)
        ans[mode] = datasets.concatenate_datasets(raw_datasets).shuffle(seed=42)
    print(ans)
    ans = ans.filter(
        lambda x: not any(
            contains_chatgpt_words(resp['value']) for resp in x['chosen'] if resp['from'] == 'gpt'
        ),
        num_proc=os.cpu_count()//2,
    )
    print(ans)
    return ans

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "",
):
    # TODO: Qwen Chat的finetune是 labels 里边包含 im_start 的
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"} #  "system": '<|im_start|>system'

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    # Apply prompt templates
    prompt_ids, prompt_targets = [], []
    answer_ids, answer_targets = [], []
    for i, source in enumerate(sources):
        if source[0]["from"]=='system':
            system_message = source[0]['value']
            source = source[1:]
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = (
            [im_start]
            + _system
            + tokenizer(system_message).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += system
        # target += (
        #     [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        # )
        target += (
            [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        )
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = (
                tokenizer(role).input_ids
                + nl_tokens
                + tokenizer(sentence["value"]).input_ids
                + [im_end]
                + nl_tokens
            )
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                    + [im_end]
                    + nl_tokens
                )
                # _target = (
                #     [IGNORE_TOKEN_ID]
                #     + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                #     + [IGNORE_TOKEN_ID]
                #     + [IGNORE_TOKEN_ID]
                # )
                prompt_ids.append(input_id[:])
                prompt_targets.append((target + _target)[:])
            elif role == "<|im_start|>assistant":
                _target = (
                    [im_start]
                    + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids)
                    + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                    + [im_end]
                    + nl_tokens
                )
                # TODO: 应该只作用于最后一句.
                # _target = (
                #     [IGNORE_TOKEN_ID]
                #     + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids)
                #     + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                #     + [im_end]
                #     + nl_tokens
                # )
                answer_ids.append(_input_id[:])
                answer_targets.append(_target[:])
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        assert len(prompt_ids[-1]) == len(prompt_targets[-1])
        assert len(answer_ids[-1]) == len(answer_targets[-1])
        # if os.getenv('DRY_RUN'):
        #     print(tokenizer.batch_decode(answer_ids))
        #     print(tokenizer.batch_decode([[ 62 if t == IGNORE_TOKEN_ID else t for t in _target] for _target in answer_targets]))

    prompt_sequence_tokens = dict(
        input_ids=prompt_ids,
        labels=prompt_targets,
        attention_mask=[
            [id != tokenizer.pad_token_id for id in ids] for ids in prompt_ids
        ],
    )
    answer_sequence_tokens = dict(
        input_ids=answer_ids,
        labels=answer_targets,
        attention_mask=[
            [id != tokenizer.pad_token_id for id in ids] for ids in answer_ids
        ],
    )
    return prompt_sequence_tokens, answer_sequence_tokens

def tokenize_row_chatml(self, feature, model = None) :
    # notice attention mask must be False or True
    batch = {}
    prompt = feature["prompt"]
    chosen = feature["chosen"]
    rejected = feature["rejected"]
    prompt_tokens, chosen_tokens = preprocess(
        [chosen], self.tokenizer, self.max_length
    )
    _, rejected_tokens = preprocess(
        [rejected], self.tokenizer, self.max_length
    )
    # for multi turn dataset, it generate multiple data item end with each resp 
    prompt_tokens = {k: v[-1] for k, v in prompt_tokens.items()}
    chosen_tokens = {k: v[-1] for k, v in chosen_tokens.items()}
    rejected_tokens = {k: v[-1] for k, v in rejected_tokens.items()}

    eos_token_id = self.tokenizer.eos_token_id
    # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
    eos_indices_prompt = [
        i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id
    ]
    # attention mask these indices to eos_token_id
    new_attention_mask = [
        False if i in eos_indices_prompt else p
        for i, p in enumerate(prompt_tokens["attention_mask"])
    ]
    prompt_tokens["attention_mask"] = new_attention_mask

    # do the same for chosen and rejected
    eos_indices_chosen = [
        i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id
    ]
    new_attention_mask_c = [
        False if i in eos_indices_chosen else p
        for i, p in enumerate(chosen_tokens["attention_mask"])
    ]
    chosen_tokens["attention_mask"] = new_attention_mask_c

    eos_indices_rejected = [
        i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id
    ]
    new_attention_mask_r = [
        False if i in eos_indices_rejected else p
        for i, p in enumerate(rejected_tokens["attention_mask"])
    ]
    rejected_tokens["attention_mask"] = new_attention_mask_r

    # TODO: add EOS token to end of answer
    chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
    chosen_tokens["attention_mask"].append(True)

    rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
    rejected_tokens["attention_mask"].append(True)

    longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    # if combined sequence is too long, truncate the prompt
    for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
        if len(answer_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
            elif self.truncation_mode == "keep_end":
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

    # if that's still too long, truncate the response
    for answer_tokens in [chosen_tokens, rejected_tokens]:
        if len(answer_tokens["input_ids"]) + longer_response_length > self.max_length:
            for k in ["input_ids", "attention_mask"]:
                answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

    # Create labels
    # TODO: hack 
    prompt_tokens['labels'] = [IGNORE_TOKEN_ID] * len(prompt_tokens['input_ids'])
    chosen_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in ["input_ids", "attention_mask", "labels"]}
    rejected_tokens = {
        k: prompt_tokens[k] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
    }
    chosen_tokens["labels"] = chosen_tokens["input_ids"][:]
    chosen_tokens["labels"][: (len(prompt_tokens["input_ids"]) + len(self.tokenizer('<|im_start|>assistant').input_ids))] = [
        IGNORE_TOKEN_ID
    ] * (len(prompt_tokens["input_ids"]) + len(self.tokenizer('<|im_start|>assistant').input_ids))
    rejected_tokens["labels"] = rejected_tokens["input_ids"][:]
    rejected_tokens["labels"][: (len(prompt_tokens["input_ids"]) + len(self.tokenizer('<|im_start|>assistant').input_ids))] = [
        IGNORE_TOKEN_ID
    ] * (len(prompt_tokens["input_ids"]) + len(self.tokenizer('<|im_start|>assistant').input_ids))
    for k, toks in {
        "chosen_": chosen_tokens,
        "rejected_": rejected_tokens,
        "": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}{type_key}"] = tokens

    if os.getenv('DRY_RUN', False):
        # print('prompt',self.tokenizer.decode(prompt_tokens['input_ids']))
        concatenated_batch = DPOTrainer.concatenated_inputs(
            batch={k: torch.tensor([v]) for k ,v in batch.items()},padding_value=self.tokenizer.pad_token_id,
        )
        print(
            json.dumps({
                'prompt-labels': self.tokenizer.decode([62 if t == -100 else t for t in prompt_tokens['labels']]),
                'chosen-labels': self.tokenizer.decode([62 if t == -100 else t for t in chosen_tokens['labels']]),
                'rejected-labels':self.tokenizer.decode([62 if t == -100 else t for t in rejected_tokens['labels']]),
                'concat-inputs': self.tokenizer.batch_decode(concatenated_batch['concatenated_input_ids']),
                'concat-labels': self.tokenizer.batch_decode([[ 62 if t == IGNORE_TOKEN_ID else t for t in _target] for _target in concatenated_batch['concatenated_labels'].tolist()]),
            })
        )
        # concatenated_batch['concatenated_labels'] * concatenated_batch['concatenated_attention_mask']
        # concatenated_batch['concatenated_input_ids'] * concatenated_batch['concatenated_attention_mask']

    return batch

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()
    training_args.hub_private_repo = True
    training_args.ddp_find_unused_parameters=False
    global preprocessing_num_workers
    preprocessing_num_workers = data_args.preprocessing_num_workers
    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args)
    raw_datasets['train'].to_json(f'{training_args.output_dir}/dpo_train.jsonl')
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    # for split in ["train", "test"]:
    #     raw_datasets[split] = raw_datasets[split].remove_columns(
    #         [col for col in raw_datasets[split].features if col not in ['prompt', 'chosen', 'rejected']]
    #     )
    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )
    if tokenizer.eos_token_id is None and 'qwen' in model_args.model_name_or_path.lower():
        tokenizer.eos_token_id = 151643 # '<|endoftext|>'
        # <|im_start|> 151644  <|im_end|> 151645
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    #####################
    # Apply chat template
    #####################
    # raw_datasets = raw_datasets.map(
    #     apply_chat_template,
    #     fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
    #     num_proc=data_args.preprocessing_num_workers,
    #     remove_columns= list(raw_datasets["train"].features)
    #     desc="Formatting comparisons with prompt template",
    # )
    # Replace column names with what TRL needs, 
    # text_chosen -> chosen  
    # text_rejected -> rejected
    # for split in ["train", "test"]:
    #     raw_datasets[split] = raw_datasets[split].rename_columns(
    #         {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    #     )

    if 'qwen' in model_args.model_name_or_path.lower():
        # no BOS token; apply chatml format
        setattr(DPOTrainer, 'tokenize_row', tokenize_row_chatml)
        setattr(DPOTrainer, 'concatenated_forward', concatenated_forward)
        # for debug:
        if os.getenv('DEBUG', False) or os.getenv('DRY_RUN',False):

            class DataDebugger:
                def __init__(self, dataset, tokenizer,max_length,max_prompt_length):
                    self.max_length = max_length
                    self.max_prompt_length = max_prompt_length
                    self.tokenizer = tokenizer
                    self.truncation_mode = 'keep_end'
                    self.dataset = dataset.map(self.tokenize_row, num_proc=1)

            setattr(DataDebugger, 'tokenize_row', tokenize_row_chatml)

            dpo_trainer = DataDebugger(
                dataset=raw_datasets['train'],
                tokenizer=tokenizer,
                max_length=training_args.max_length,
                max_prompt_length=training_args.max_prompt_length,
            )
            return

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = None
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # For consistency with model weights, we use the same value as `torch_dtype` which is float16 for PEFT models
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision):
        # load the model, merge the adapter weights and unload the adapter
        # Note: to run QLora, you will need to merge the based model separately as the merged model in 16bit
        logger.info(f"Merging peft adapters for {model_args.model_name_or_path=}")

        peft_config = PeftConfig.from_pretrained(
            model_args.model_name_or_path, 
            revision=model_args.model_revision
        )

        model_kwargs = dict(
            revision=model_args.base_model_revision,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            trust_remote_code=True
        )
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     peft_config.base_model_name_or_path,
        #     **model_kwargs,
        # )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map={'': int(os.getenv('LOCAL_RANK',0))},
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model, 
            model_args.model_name_or_path, 
            revision=model_args.model_revision
        )
        model.eval()
        model = model.merge_and_unload(progressbar=True)
        model_kwargs = None
    else:
        # BUG: 直接送模型name给trl dpotrainer初始化有梯度错误，暂未查出原因
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            device_map={'': int(os.getenv('LOCAL_RANK',0))},
            revision=model_args.model_revision,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            quantization_config=quantization_config,
            trust_remote_code=True
        )         

    ref_model = model
    # ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=model_args.lora_target_modules,
            modules_to_save=model_args.lora_modules_to_save,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    #########################
    # Instantiate DPO trainer
    #########################
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        # model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        truncation_mode="keep_end",
        peft_config=peft_config if model_args.use_peft else None,
        # gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
    )
    if model_args.use_peft is True and 'qwen' in model_args.model_name_or_path.lower():
        # for name, module in model.named_modules():
        #     if "ln_1" in name or 'ln_2' in name or 'ln_f' in name:
        #         module.to(torch_dtype)
        #     if "lm_head" in name or "wte" in name:
        #         if hasattr(module, "weight"):
        #             module.to(torch_dtype)     
        for n,p in model.named_parameters():
            if p.dtype == torch.float32 and 'lora' not in n:
                p.data = p.data.to(torch_dtype)  
        print(([ (n,p.dtype) for n,p in model.named_parameters() ]))
    if os.getenv('DEBUG', False):
        logger.info([ (n,p.dtype) for n,p in dpo_trainer.model.named_parameters() ])
        logger.info([ n for n,p in dpo_trainer.model.named_parameters() if p.requires_grad ])
        requires_grad = []
        for name, param in dpo_trainer.model.named_parameters(recurse=True):
            if param.requires_grad:
                requires_grad.append(f"{name}: {param.requires_grad}")
        if len(requires_grad) == 0:
            logger.error("there are no parameters that require gradient updates")
    # no use 
    delattr(dpo_trainer.model.model.base_model.__class__.__bases__[0], '_set_gradient_checkpointing')
    ###############
    # Training loop
    ###############
    train_result = dpo_trainer.train(resume_from_checkpoint=True)
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()

    logger.info("*** Training complete ***")

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = dpo_trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(raw_datasets["test"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(raw_datasets["test"]))
        dpo_trainer.log_metrics("eval", metrics)
        dpo_trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    dpo_trainer.save_model(training_args.output_dir)
    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        dpo_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        dpo_trainer.model.config.use_cache = True
        dpo_trainer.model.config.save_pretrained(training_args.output_dir)
        if training_args.push_to_hub is True:
            dpo_trainer.push_to_hub()

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    try:
        main()
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
    