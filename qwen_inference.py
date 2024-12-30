import math
import os
import argparse
import json
import torch
from tqdm import tqdm
import numpy as np
import cv2
import base64
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoConfig

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.mm_utils import (
    process_anyres_image,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.model.builder import load_pretrained_model
from llava.train.train import smart_tokenizer_and_embedding_resize


class VideoProcessor:
    def __init__(self, args):
        self.args = args

    def load_video(self, video_path):
        """Load and process video frames."""
        if self.args.frame_count == 0:
            return np.zeros((1, 336, 336, 3))

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        fps = round(vr.get_avg_fps())
        video_duration = total_frames / fps

        frame_indices = [i for i in range(0, total_frames, fps)]
        frame_times = [i / fps for i in frame_indices]

        if len(frame_indices) > self.args.frame_count or self.args.force_sample:
            uniform_samples = np.linspace(
                0, total_frames - 1, self.args.frame_count, dtype=int
            )
            frame_indices = uniform_samples.tolist()
            frame_times = [i / fps for i in frame_indices]

        frame_times_str = ",".join([f"{t:.2f}s" for t in frame_times])
        frames = vr.get_batch(frame_indices).asnumpy()

        return frames, frame_times_str, video_duration

    @staticmethod
    def load_video_base64(path):
        """Convert video frames to base64 encoding."""
        video = cv2.VideoCapture(path)
        base64_frames = []

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()
        return base64_frames


class VideoInference:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_response(self, video, prompt):
        """Generate model response for video input."""
        if self.model.config.mm_use_im_start_end:
            prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{prompt}"
        else:
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )

        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                self.tokenizer.pad_token_id = 151643

        attention_mask = (
            input_ids.ne(self.tokenizer.pad_token_id).long().to(self.device)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=video,
                    attention_mask=attention_mask,
                    modalities="video",
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=256,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

                generated_text = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()

                if generated_text.endswith(stop_str):
                    generated_text = generated_text[: -len(stop_str)].strip()

                return generated_text

        except Exception as e:
            print(f"Generation error: {str(e)}")
            return "Can you describe another aspect of the video?"


ANSWER_PROMPTS = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": "",
}


def setup_model(model_path, args):
    """Setup the model, tokenizer and processors."""
    model_name = get_model_name_from_path(model_path)

    if args.overwrite:
        config = {
            "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
            "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
            "mm_newline_position": args.mm_newline_position,
        }

        cfg_pretrained = AutoConfig.from_pretrained(model_path)

        if "qwen" not in model_path.lower():
            if "224" in cfg_pretrained.mm_vision_tower:
                min_tokens = (
                    args.frame_count * (16 // args.mm_spatial_pool_stride) ** 2 + 1000
                )
            else:
                min_tokens = (
                    args.frame_count * (24 // args.mm_spatial_pool_stride) ** 2 + 1000
                )

            scaling = math.ceil(min_tokens / 4096)
            if scaling >= 2:
                if "vicuna" in cfg_pretrained._name_or_path.lower():
                    config["rope_scaling"] = {
                        "factor": float(scaling),
                        "type": "linear",
                    }
                config["max_sequence_length"] = 4096 * scaling
                config["tokenizer_model_max_length"] = 4096 * scaling

        return load_pretrained_model(
            model_path, args.model_base, model_name, overwrite_config=config
        )

    return load_pretrained_model(model_path, args.model_base, model_name)


def main():
    parser = argparse.ArgumentParser(description="Video LLM Processing")
    parser.add_argument(
        "--video_dir", required=True, help="Directory containing video files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory for output predictions"
    )
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument(
        "--questions_dir", required=True, help="Directory containing question files"
    )
    parser.add_argument(
        "--task_type",
        default="multi-choice",
        choices=["multi-choice", "captioning", "caption_matching", "yes_no"],
    )
    parser.add_argument("--frame_count", type=int, default=4)
    parser.add_argument(
        "--overwrite", type=lambda x: str(x).lower() == "true", default=True
    )
    parser.add_argument(
        "--force_sample", type=lambda x: str(x).lower() == "true", default=False
    )
    parser.add_argument("--model_base", default=None)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup model and processors
    tokenizer, model, image_processor, context_len = setup_model(args.model_path, args)
    model = model.to("cuda")

    # Load questions
    question_file = os.path.join(args.questions_dir, f"{args.task_type}.json")
    with open(question_file, "r") as f:
        questions = json.load(f)

    # Initialize or load predictions
    pred_file = os.path.join(args.output_dir, f"{args.task_type}.json")
    if os.path.isfile(pred_file):
        with open(pred_file, "r") as f:
            predictions = json.load(f)
    else:
        predictions = {}

    # Setup processors
    video_processor = VideoProcessor(args)
    inference_engine = VideoInference(model, tokenizer)

    # Process videos
    for video_id, data in tqdm(questions.items()):
        if video_id not in predictions:
            predictions[video_id] = {}
            video_path = os.path.join(args.video_dir, f"{video_id}.mp4")

            for dimension, question_list in data.items():
                predictions[video_id][dimension] = []
                for question in question_list:
                    prompt = question["question"] + ANSWER_PROMPTS[args.task_type]
                    video, _, _ = video_processor.load_video(video_path)
                    video = (
                        image_processor.preprocess(video, return_tensors="pt")[
                            "pixel_values"
                        ]
                        .half()
                        .cuda()
                    )
                    video = [video]

                    prediction = inference_engine.generate_response(video, prompt)
                    predictions[video_id][dimension].append(
                        {
                            "question": question["question"],
                            "answer": question["answer"],
                            "prediction": prediction,
                        }
                    )

            # Save predictions after each video
            with open(pred_file, "w") as f:
                json.dump(predictions, f, indent=4)


if __name__ == "__main__":
    main()
