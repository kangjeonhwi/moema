#!/bin/bash
PROJECT_ROOT="/home/MoE/moema2/moema"

# ❗ PYTHONPATH에 프로젝트 루트를 추가하여 파이썬이 src 폴더를 찾게 함
export PYTHONPATH="$PROJECT_ROOT"

# WandB 설정을 위한 환경 변수
export WANDB_PROJECT="moe_instruction_tuning_final"
export WANDB_NAME="llama3.2-3b-moe-sft-frozen-$(date +%Y%m%d-%H%M)"

export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=INFO


# 사용할 GPU 개수
NUM_GPUS=8

# 모델 및 출력 경로
# DeepSpeed 실행
# [수정됨] 문제가 발생한 --evaluation_strategy 인자 삭제
uv run deepspeed --num_gpus=$NUM_GPUS "${PROJECT_ROOT}/Exp/instruction_tuning.py" \
