import argparse
import yaml
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.modeling_llama import LlamixForCausalLM
import torch
from tqdm import tqdm
from datasets import load_dataset
from accelerate import Accelerator

import evaluate

def calculate_exact_match(predictions, references):
    em_scores = []
    for pred, ref_list in zip(predictions, references):
        # references는 보통 여러 개의 정답을 포함할 수 있으므로, 하나라도 일치하면 EM으로 간주
        em = 1 if pred.strip() in [r.strip() for r in ref_list] else 0
        em_scores.append(em)
    return sum(em_scores) / len(em_scores) if em_scores else 0

def calculate_f1_score(predictions, references):

    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        import re
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            return re.sub(r'[^\w\s]', '', text)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    f1_scores = []
    for pred, ref_list in zip(predictions, references):
        pred_tokens = normalize_answer(pred).split()
        max_f1 = 0
        for ref in ref_list:
            ref_tokens = normalize_answer(ref).split()
            common_tokens = set(pred_tokens) & set(ref_tokens)

            if len(common_tokens) == 0:
                f1 = 0
            else:
                precision = len(common_tokens) / len(pred_tokens)
                recall = len(common_tokens) / len(ref_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
            max_f1 = max(max_f1, f1)
        f1_scores.append(max_f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0


def evaluate_model_qa(model_path: str,llama_path:str, config_path: str, batch_size: int = 8, device: str = None):
    accelerator = Accelerator()

    if device is None:
        device = accelerator.device

    print(f"Loading model from {model_path}...")
    model = LlamixForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(llama_path)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.chat_template is None:
        print("Tokenizer does not have a chat template. Using a default template.")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}\n\n"
            "{% elif message['role'] == 'user' %}"
            "USER: {{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}"
            "ASSISTANT: {{ message['content'] }}{% endif %}"
            "{% if loop.last and message['role'] == 'user' %}ASSISTANT:{% endif %}" 
            "{% endfor %}"
        )

    model, tokenizer = accelerator.prepare(model, tokenizer)
    model.eval()

    print(f"Loading evaluation config from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    all_results = {}

    for dataset_cfg in config['datasets']:
        dataset_name = dataset_cfg['name']
        dataset_path = dataset_cfg['path']
        split = dataset_cfg['split']
        system_key = dataset_cfg.get('system_key', None)
        question_key = dataset_cfg['question_key']
        context_key = dataset_cfg.get('context_key', None)
        answers_key = dataset_cfg['answers_key']

        print(f"\n--- Evaluating on dataset: {dataset_name} ---")
        try:
            dataset = load_dataset('json', data_files=dataset_path, split=split)
            print(f"Loaded {len(dataset)} samples from {dataset_name}.")
        except Exception as e:
            print(f"Error loading dataset {dataset_name} from {dataset_path}: {e}")
            continue

        predictions = []
        references = []

        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Generating responses for {dataset_name}"):
            batch = dataset[i : i + batch_size]

            batch_messages = []
            for j in range(len(batch[question_key])):
                system_prompt_content = batch[system_key][j] if system_key and batch[system_key][j] else ""
                question = batch[question_key][j]
                context = batch[context_key][j] if context_key and batch[context_key][j] else ""

                messages = []
                if system_prompt_content:
                    messages.append({"role": "system", "content": system_prompt_content})

                user_content = ""
                if context:
                    user_content += f"Context: {context}\n\n"
                user_content += f"Question: {question}"
                messages.append({"role": "user", "content": user_content})
                batch_messages.append(messages)

            try:
                batch_formatted_prompts = [
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    for messages in batch_messages
                ]
            except Exception as e:
                print(f"Error applying chat template: {e}. Please check your tokenizer's chat_template.")
                continue

            inputs = tokenizer(batch_formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256, # 생성할 최대 토큰 수
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.2
                )

            for k in range(len(inputs.input_ids)):
                generated_text = tokenizer.decode(outputs[k, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                predictions.append(generated_text.strip())
                references.append(batch[answers_key][k])

        em_score = calculate_exact_match(predictions, references)
        f1_score = calculate_f1_score(predictions, references)

        rouge_metric = evaluate.load("rouge")
        rouge_results = rouge_metric.compute(predictions=predictions, references=references)

        dataset_results = {
            "exact_match": em_score,
            "f1_score": f1_score,
            "rouge": rouge_results
        }
        all_results[dataset_name] = dataset_results
        print(f"Results for {dataset_name}:")
        for metric, value in dataset_results.items():
            print(f"  {metric}: {value}")

    print("\n--- Overall Evaluation Results ---")
    for dataset, results in all_results.items():
        print(f"Dataset: {dataset}")
        for metric, value in results.items():
            print(f"  {metric}: {value}")

    output_filename = "qa_evaluation_results_chat_template.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"\nEvaluation results saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA performance of a language model.")
    parser.add_argument("--model_path", type=str, default = "/home/MoE/moema2/moema/output/model_ckpt/llamix-3.2-3b-instruct-finetuning/checkpoint-400",
                        help="Path to the pre-trained language model (e.g., Hugging Face model path).")
    parser.add_argument("--llama_path", type=str, default = "meta-llama/Llama-3.2-3B-Instruct",
                        help="Path to the pre-trained language model (e.g., Hugging Face model path).")
    parser.add_argument("--config_path", type=str, default="./default_config.yaml",
                        help="Path to the evaluation dataset configuration YAML file.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for model inference.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to use for inference (e.g., 'cuda', 'cpu'). Defaults to Accelerator's choice.")

    args = parser.parse_args()

    evaluate_model_qa(
        model_path=args.model_path,
        llama_path = args.llama_path,
        config_path=args.config_path,
        batch_size=args.batch_size,
        device=args.device
    )