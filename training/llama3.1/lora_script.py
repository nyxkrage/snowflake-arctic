# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based off the lora script here: https://github.com/artidoro/qlora/blob/main/qlora.py
import copy
import json
import os
import deepspeed
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging

import torch
import transformers
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel
import argparse
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
)
from datasets import load_dataset
from datasets.combine import concatenate_datasets

import deepspeed.comm as dist
from deepspeed.linear import LoRAConfig, QuantizationConfig

from axo_dataset import ShareGPTPrompterV2, SimpleShareGPTPromptTokenizingStrategy, TokenizedPromptDataset, add_get_turns_to_conversation, check_example_labels
from unsloth_grad import hf_grad_checkpoint_unsloth_wrapper

transformers.modeling_utils.checkpoint = hf_grad_checkpoint_unsloth_wrapper

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

add_get_turns_to_conversation()
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Meta-Llama-3.1-405B"
    )
    tokenizer_name_or_path: Optional[str] = field(
        default="meta-llama/Meta-Llama-3.1-405B"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )

@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    datasets: str = field(
        default='alpaca',
        metadata={"help": "Which datasets to finetune on. Comma separated."}
    )
    conversation_format: Optional[str] = field(
        default="claude",
        metadata={"help": "Which conversation format is used."}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )

    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    quantize: bool = field(
        default=False,
        metadata={"help": "quantize frozen base weights or not."}
    )
    bits: int = field(
        default=8,
        metadata={"help": "How many bits to use for quantization."}
    )
    base_weight_sharding: bool = field(
        default=False,
        metadata={"help": "Shard base weights with DP world size, similar to ZeRO-3."}
    )
    offload: bool = field(
        default=False,
        metadata={"help": "Offload the base weights to CPU."}
    )
    offload_ratio: float = field(
        default=0.0,
        metadata={"help": "Fraction of base weights to offload to CPU."}
    )

    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=1.0, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    max_sequence_len: float = field(default=8192, metadata={"help": 'Max sequence lenth for training. Increase if needed but longer sequences take more memory.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    activation_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})

    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='epoch', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})


def get_accelerate_model(args):
    if args.full_finetune: assert args.bits in [16, 32]

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    _apply_liger_kernel(model_config.model_type)
    

    print(f'loading base model {args.model_name_or_path}...')

    if not args.full_finetune:
        base_weight_shards = dist.get_world_size() if args.base_weight_sharding else 1
        lora_config = LoRAConfig(
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            base_weight_sharding=base_weight_shards,
            offload=args.offload,
            offload_ratio=args.offload_ratio,
            target_mods=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        )
    else:
        lora_config = None

    if args.quantize:
        assert args.bits == 8, "currently deepspeed only supports fp8 for llama"
        quantization_config = QuantizationConfig(q_bits=args.bits)
    else:
        quantization_config = None
 
    # Tokenizer
    print(f"Loading tokenizer {args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id=tokenizer.eos_token_id
        tokenizer.pad_token=tokenizer.eos_token

    print(f'loading base model {args.model_name_or_path}...')
    with deepspeed.linear.Init(lora_config=lora_config, quant_config=quantization_config):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            attn_implementation="flash_attention_2",
        )

    print("created model")

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False  # turn off when gradient checkpointing is enabled

    return model, tokenizer


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> List[Dict]:
    """
    Make dataset for supervised fine-tuning.
    """
    def load_data(dataset_names: str):
        return concatenate_datasets([load_dataset(dataset)["train"] for dataset in dataset_names.split(",")])
    
    dataset = load_data(args.datasets)

    strategy = SimpleShareGPTPromptTokenizingStrategy(
        ShareGPTPrompterV2(
            conversation=args.conversation_format,
            role_key_model=None,
            role_key_human=None,
        ),
        tokenizer,
        False,  # train_on_inputs
        args.max_sequence_len,  # sequence_len
    )

    train_dataset = TokenizedPromptDataset(
        strategy, dataset, process_count=1
    )  

    print("!! sample values from the data !!")
    for example in train_dataset.select(range(5)):
        print(check_example_labels(example, tokenizer, True))

    exit()

    return train_dataset

def train():
    dist.init_distributed()
    torch.autograd.set_detect_anomaly(True)
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    model, tokenizer = get_accelerate_model(args)

    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    train_dataset = make_data_module(tokenizer=tokenizer, args=args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.model.train()

    print_trainable_parameters(args, model)

    all_metrics = {"run_name": args.run_name}
    # Training
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    all_metrics.update(metrics)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
        fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
