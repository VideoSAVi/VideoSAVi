#!/usr/bin/env python3

import argparse
import json
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    transform = T.Compose(
        [T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )
    return transform


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    # calculate the target width and height
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_video(video_path, bound=None, input_size=448, max_num=12, num_segments=8):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Script")

    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--model_path", type=str, default="yogkul2000/VideoSAVi", help="Path to the VideoSAVi model")

    parser.add_argument("--num_segments", type=int, default=8, help="Number of video segments to sample (default: 8)")

    parser.add_argument("--max_patches", type=int, default=12, help="Maximum patches per frame (default: 12)")

    parser.add_argument("--input_size", type=int, default=448, help="Input image size (default: 448)")

    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate (default: 1024)")

    parser.add_argument("--do_sample", action="store_true", default=False, help="Whether to use sampling for generation")

    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature)")

    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter (default: 1.0)")

    parser.add_argument("--question", type=str, default="What is happening in this video?", help="Question to ask about the video")

    parser.add_argument("--output_file", type=str, default=None, help="Optional output file to save results")

    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference (default: cuda)")

    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Torch dtype for model (default: bfloat16)")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument("--no_follow_up", action="store_true", help="Skip follow-up question")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        print(f"Loading model from: {args.model_path}")
        print(f"Processing video: {args.video_path}")
        print(f"Video segments: {args.num_segments}")
        print(f"Max patches per frame: {args.max_patches}")

    torch_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = torch_dtype_map[args.torch_dtype]

    try:
        model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval()

        if args.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

        if args.verbose:
            print("Model and tokenizer loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        if args.verbose:
            print("Loading and processing video...")

        pixel_values, num_patches_list = load_video(args.video_path, num_segments=args.num_segments, max_num=args.max_patches, input_size=args.input_size)

        pixel_values = pixel_values.to(torch_dtype)
        if args.device == "cuda" and torch.cuda.is_available():
            pixel_values = pixel_values.cuda()

        if args.verbose:
            print(f"Video processed: {len(num_patches_list)} segments, {pixel_values.shape[0]} total patches")

    except Exception as e:
        print(f"Error processing video: {e}")
        return

    # Create video prefix for frames
    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])

    # Generation config
    generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": args.do_sample, "temperature": args.temperature, "top_p": args.top_p}

    results = {}

    try:
        question = video_prefix + args.question

        if args.verbose:
            print(f"\nAsking question: {args.question}")

        response, history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)

        print(f"\nUser: {args.question}")
        print(f"VideoSAVi: {response}")

        results["question_1"] = {"question": args.question, "response": response}

        # Clear GPU cache
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during first inference: {e}")
        return

    # Save results if output file specified
    if args.output_file:
        try:
            results["video_path"] = args.video_path
            results["model_path"] = args.model_path
            results["config"] = {"num_segments": args.num_segments, "max_patches": args.max_patches, "input_size": args.input_size, "generation_config": generation_config}

            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\nResults saved to: {args.output_file}")

        except Exception as e:
            print(f"Error saving results: {e}")

    if args.verbose:
        print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
