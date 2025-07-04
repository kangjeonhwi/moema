# scripts/run_continued_pretraining.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from src.model.modeling_moema import MoemaForCausalLM 

def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

def main():
    print("Llama loading...")
    llama_path = "meta-llama/Llama-3.2-1B" # Base llama
    DTYPE = torch.bfloat16

    print("Model processing...")
    model = MoemaForCausalLM.from_llama(
        model_path=llama_path,
        num_experts=8,
        num_experts_per_tok=2,
        torch_dtype=DTYPE,
    )

    tokenizer = AutoTokenizer.from_pretrained(llama_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nWeight Freezing...")
    for param in model.parameters():
        param.requires_grad = False

    trainable_layer_keywords = ["block_sparse_moe", "gate"]
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_layer_keywords):
            param.requires_grad = True

    print("\n✅ 최종 학습 가능 파라미터 현황:")
    print_trainable_parameters(model)


    # --- 3. 데이터셋 준비 (Continued Pre-training용) ---
    # Instruction 형식이 아닌, 순수 텍스트 데이터셋을 로드합니다.
    train_file = "dataset/train/llama_finetune_2048.jsonl" # 파일명은 그대로 사용
    dataset = load_dataset("json", data_files=train_file, split="train")

    # 텍스트 데이터를 토큰 ID로 변환하는 전처리 함수
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


    # --- 4. 학습 설정 (TrainingArguments) ---
    training_args = TrainingArguments(
        output_dir="./outputs/domain_pretrain",
        per_device_train_batch_size=2, # GPU 메모리에 맞춰 조정
        gradient_accumulation_steps=4,
        learning_rate=5e-5, # 사전학습은 일반적으로 더 낮은 학습률을 사용
        num_train_epochs=1,
        save_strategy="epoch",
        logging_steps=10,
        bf16=True,
        report_to="tensorboard",
    )
    
    # --- 5. 기본 'Trainer' 생성 및 학습 시작 ---
    # SFTTrainer 대신 기본 Trainer를 사용합니다.
    # DataCollatorForLanguageModeling은 텍스트를 모델이 학습하기 좋은 형태로 묶어줍니다.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n🚀 도메인 적응 사전학습(Continued Pre-training)을 시작합니다!")
    trainer.train()

    # --- 6. 학습된 모델 저장 ---
    final_model_path = "./outputs/domain_pretrain/final_model"
    trainer.save_model(final_model_path)
    print(f"\n🎉 학습 완료! 법률 지식이 주입된 모델이 {final_model_path}에 저장되었습니다.")


if __name__ == "__main__":
    main()
