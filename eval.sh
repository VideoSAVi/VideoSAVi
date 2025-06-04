export DECORD_EOF_RETRY_MAX=20480
export OPENAI_API_KEY=""
MODEL_NAME=internvl2
CKPT_PATH=""

echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
MASTER_PORT=$((18000 + $RANDOM % 100))
NUM_GPUS=4
MAX_NUM_FRAMES=32

MODALITY=video

mkdir -p ./logs_internvl_iter4_savi
BENCHMARKS=(
"perceptiontest_val_mc" 
"mvbench"
"egoschema_subset"
"nextqa_mc_test"
"longvideobench_val_v"
)

# Loop through each benchmark
for TASK in "${BENCHMARKS[@]}"; do
    echo "Evaluating benchmark: $TASK"
    
    JOB_NAME="eval_${TASK}_internvl_iter4_savi_$(date +"%Y%m%d_%H%M%S")"
    
    # Random port to avoid conflicts
    MASTER_PORT=$((18000 + $RANDOM % 100))
    
    accelerate launch --num_processes ${NUM_GPUS} --main_process_port ${MASTER_PORT} -m lmms_eval \
        --model ${MODEL_NAME} \
        --model_args pretrained=$CKPT_PATH,num_frame=$MAX_NUM_FRAMES,modality=$MODALITY \
        --tasks $TASK \
        --batch_size 1 \
        --output_path ./logs/${JOB_NAME}_${MODEL_NAME}_f${MAX_NUM_FRAMES}

done

echo "All benchmark evaluations completed!" 