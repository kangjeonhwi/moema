import os
import sys
import torch
import logging
import nltk
import numpy as np
import evaluate

import wandb
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    set_seed,
    TrainerCallback, TrainerState, TrainerControl
)
from trl import SFTTrainer, SFTConfig
from src.model.modeling_llama import LlamixForCausalLM
from rouge_score import rouge_scorer

# --- 1. 기본 설정 ---
logger = logging.getLogger(__name__)
os.environ["WANDB_PROJECT"] = "moe-finetuning-project"

nltk.download('punkt_tab')
nltk.download("punkt", quiet=True)

rouge = evaluate.load("rouge")
tokenizer = None
import torch.distributed as dist
import datetime

dist.init_process_group(backend="nccl", timeout=datetime.timedelta(days=2))

def is_main_process():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def apply_chat_template(sample, tokenizer):
    messages = []
    if 'system' in sample and sample['system'] and sample['system'].strip():
        messages.append({"role": "system", "content": sample['system']})
 
    user_content = sample['instruction']
    if 'input' in sample and sample['input'] and sample['input'].strip():
        user_content += f"\n\n{sample['input']}"
    
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": sample['output']})
    formatted_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": formatted_text}

def apply_chat_template_with_global_tokenizer(sample):
    global tokenizer
    return apply_chat_template(sample, tokenizer)

def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"학습 가능 파라미터: {trainable_params} | 전체 파라미터: {all_param} | 학습 비율(%): {100 * trainable_params / all_param:.2f}")

def build_compute_metrics(tokenizer):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)


        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        scores = {"rouge1": [], "rougeL": []}
        for pred, label in zip(decoded_preds, decoded_labels):
            result = scorer.score(label, pred)
            scores["rouge1"].append(result["rouge1"].fmeasure)
            scores["rougeL"].append(result["rougeL"].fmeasure)

        return {
            "rouge1": round(np.mean(scores["rouge1"]), 4),
            "rougeL": round(np.mean(scores["rougeL"]), 4),
        }
    return compute_metrics

@dataclass
class DataArguments:
    train_files_en: str = field(default="/home/MoE/moema2/moema/dataset/processed_dataset/en/train.jsonl")
    train_files_ko: str = field(default="/home/MoE/moema2/moema/dataset/processed_dataset/ko/train.jsonl")
    train_files_legal: str = field(default="/home/MoE/moema2/moema/dataset/processed_dataset/legal/train.jsonl")
    eval_files_en: str = field(default="/home/MoE/moema2/moema/dataset/processed_dataset/en/eval.jsonl")
    eval_files_ko: str = field(default="/home/MoE/moema2/moema/dataset/processed_dataset/ko/eval.jsonl")
    eval_files_legal: str = field(default="/home/MoE/moema2/moema/dataset/processed_dataset/legal/eval.jsonl")
    llamix_path: str = field(default="/home/MoE/moema2/moema/output/twined_models/Llama-3.2-3B-Instruct_drop_0.5")

def main():
    training_args = SFTConfig(
        run_name="llamix-3.2-3b-instruct-finetuning",
        output_dir="/home/MoE/moema2/moema/output/model_ckpt/llamix-3.2-3b-0.5-instruct-finetuning",
        overwrite_output_dir=True,
        report_to="wandb",
        seed=42,
        packing=True,
        max_seq_length=2048,
        dataset_text_field="text",
        deepspeed="/home/MoE/moema2/moema/configs/instruct/ds_config.json",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="steps",        
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=5,
        bf16=True,
        ddp_find_unused_parameters=False,
    )
    data_args = DataArguments()
    set_seed(training_args.seed)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset_en = load_dataset("json", data_files=data_args.train_files_en, split="train")
    train_dataset_ko = load_dataset("json", data_files=data_args.train_files_ko, split="train")
    train_dataset_legal = load_dataset("json", data_files=data_args.train_files_legal, split="train")

    ''' # Sample half of the English data
    num_en_samples = len(train_dataset_en) // 2
    train_dataset_en_sampled = train_dataset_en.shuffle(seed=42).select(range(num_en_samples))

    # Sample half of the Korean data
    num_ko_samples = len(train_dataset_ko) // 2
    train_dataset_ko_sampled = train_dataset_ko.shuffle(seed=42).select(range(num_ko_samples))

    '''
    train_dataset = concatenate_datasets([
        train_dataset_en,
        train_dataset_ko,
        train_dataset_legal
    ])

    train_dataset = train_dataset.map(
        apply_chat_template_with_global_tokenizer,
        remove_columns=train_dataset.column_names,
        num_proc=os.cpu_count()
    )

    # 평가 데이터셋 로드 (이전과 동일)
    eval_dataset = load_dataset("json", data_files=data_args.eval_files_ko, split = "train")
    eval_dataset = eval_dataset.shuffle(seed=training_args.seed).select(range(min(100, len(eval_dataset))))
    eval_dataset = eval_dataset.map(
        apply_chat_template_with_global_tokenizer,
        remove_columns=eval_dataset.column_names
    )

    model = LlamixForCausalLM.from_pretrained(data_args.llamix_path, torch_dtype=torch.bfloat16)
    model.config.attn_implementation = "flash_attention_2"
    for name, param in model.named_parameters():
        param.requires_grad = (".mlp2." in name or ".alpha" in name)
    
    print_trainable_parameters(model)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    logger.info("학습 시작")
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    logger.info("학습 완료")

if __name__ == "__main__":
    main()
