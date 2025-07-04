# 실험하고 싶은 전문가(expert)의 수를 공백으로 구분하여 나열합니다.
EXPERTS_LIST="2"
SOURCE_MODEL="meta-llama/Llama-3.2-1B"


UPCYCLING_METHOD="naive"
NUM_EXPERTS_PER_TOK=1
FFN_INIT_RATIO=0.25
SEED=42
BASE_OUTPUT_PATH="/home/MoE/moema2/moema/output/upcycled_models"


# --- 스크립트 메인 로직 ---
echo "Starting MoE model upcycling experiments..."
echo "============================================="
for num_experts in $EXPERTS_LIST
do
  # 각 실험에 대한 출력 경로를 동적으로 설정합니다.
  # 예: ./output/Llama-3.2-1B-MoE-2e
  OUTPUT_PATH="${BASE_OUTPUT_PATH}/${SOURCE_MODEL}-${UPCYCLING_METHOD}-${num_experts}e"

  echo ""
  echo ">>> [START] Running experiment for ${num_experts} experts."
  echo ">>> Output will be saved to: ${OUTPUT_PATH}"

  # uv run을 사용하여 업사이클링 스크립트 실행
  uv run python -m src.upcycling_strategies.unified_upcycling \
    --source_model_path "$SOURCE_MODEL" \
    --output_path "$OUTPUT_PATH" \
    --upcycling_method "$UPCYCLING_METHOD" \
    --num_experts "$num_experts" \
    --num_experts_per_tok "$NUM_EXPERTS_PER_TOK" \
    --ffn_init_ratio "$FFN_INIT_RATIO" \
    --seed "$SEED"

  # 작업이 성공적으로 끝났는지 확인 (선택 사항)
  if [ $? -eq 0 ]; then
    echo ">>> [SUCCESS] Finished experiment for ${num_experts} experts."
  else
    echo ">>> [ERROR] Experiment for ${num_experts} experts failed."
    exit 1
  fi
  echo "---------------------------------------------"
done

echo ""
echo "All conversion have been completed."
echo "============================================="