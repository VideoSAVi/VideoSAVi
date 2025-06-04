#!/bin/bash

VIDEO_PATH=""
MODEL_PATH="/scratch/ykulka10/VideoSAVi"
QUESTION="What is happening in this video?"
OUTPUT_DIR="./inference_results"
NUM_SEGMENTS=8
MAX_PATCHES=12

while [[ $# -gt 0 ]]; do
    case $1 in
        --video_path)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --question)
            QUESTION="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_segments)
            NUM_SEGMENTS="$2"
            shift 2
            ;;
        --max_patches)
            MAX_PATCHES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done


if [ -z "$VIDEO_PATH" ]; then
    echo "Error: --video_path is required"
    echo "Usage: $0 --video_path  [options]"
    echo "Options:"
    echo "  --video_path        Path to input video (required)"
    echo "  --model_path        Model path (default: yogkul2000/VideoSAVi)"
    echo "  --question          Question to ask (default: 'What is happening in this video?')"
    echo "  --output_dir        Output directory (default: ./inference_results)"
    echo "  --num_segments       Number of video segments (default: 8)"
    echo "  --max_patches        Max patches per frame (default: 12)"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/${VIDEO_NAME}_${TIMESTAMP}.json"

python infer.py \
    --video_path "$VIDEO_PATH" \
    --model_path "$MODEL_PATH" \
    --question "$QUESTION" \
    --output_file "$OUTPUT_FILE" \
    --num_segments $NUM_SEGMENTS \
    --max_patches $MAX_PATCHES \
    --verbose
