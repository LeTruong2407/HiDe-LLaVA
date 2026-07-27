from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from llava import conversation as conversation_lib
from llava.model.multimodal_encoder.clip_encoder import (
    CLIPTextTower,
    CLIPVisionTower,
)
from llava.train.train_MOE import (
    DataArguments,
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
)


STATISTIC_NAMES = (
    "image_anchors",
    "text_anchors",
    "image_boundary",
    "text_boundary",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild HiDe routing anchors without retraining."
    )
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--previous-checkpoint-dir")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--model-max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_clip_prompts(input_ids, tokenizer):
    input_pad = np.where(
        input_ids.cpu().numpy() != -200,
        input_ids.cpu().numpy(),
        tokenizer.pad_token_id,
    )
    decoded_inputs = tokenizer.batch_decode(
        input_pad,
        skip_special_tokens=True,
    )
    decoded_hidden_inputs = [
        "\n".join(decoded_input.split("\n")[1:])
        for decoded_input in decoded_inputs
    ]
    return [
        decoded_input.split(" ASSISTANT")[0]
        for decoded_input in decoded_hidden_inputs
    ]


def accumulate_rows(features, running_sum, count):
    rows = features.detach().reshape(-1, features.shape[-1]).float().cpu()
    rows = rows[torch.isfinite(rows).all(dim=1)]
    if rows.numel() == 0:
        return running_sum, count
    if running_sum is None:
        running_sum = torch.zeros(
            rows.shape[-1],
            dtype=torch.float64,
        )
    running_sum.add_(rows.double().sum(dim=0))
    return running_sum, count + rows.shape[0]


def find_statistic_key(state, statistic_name, task_id):
    suffix = f"{statistic_name}.{task_id}"
    matches = [key for key in state if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one checkpoint key ending in {suffix!r}; "
            f"found {matches}"
        )
    return matches[0]


def synchronize_previous_statistics(state, source_state, task_id):
    synchronized = []
    for previous_task_id in range(task_id):
        for statistic_name in STATISTIC_NAMES:
            target_key = find_statistic_key(
                state,
                statistic_name,
                previous_task_id,
            )
            source_key = find_statistic_key(
                source_state,
                statistic_name,
                previous_task_id,
            )
            source_value = source_state[source_key].float()
            if not torch.isfinite(source_value).all():
                raise RuntimeError(
                    f"Previous checkpoint statistic {source_key} "
                    "contains non-finite values"
                )
            state[target_key] = source_value.clone()
        synchronized.append(previous_task_id)
    return synchronized


def validate_rebuilt_state(
    state,
    image_anchor_key,
    text_anchor_key,
    image_boundary_key,
    text_boundary_key,
    image_count,
    text_count,
):
    for key in (image_anchor_key, text_anchor_key):
        if state[key].dtype != torch.float32:
            raise RuntimeError(f"{key} was saved as {state[key].dtype}")
        if not torch.isfinite(state[key]).all():
            raise RuntimeError(f"{key} contains non-finite values")
    if state[image_boundary_key].dtype != torch.float32:
        raise RuntimeError(
            f"{image_boundary_key} was saved as "
            f"{state[image_boundary_key].dtype}"
        )
    if state[text_boundary_key].dtype != torch.float32:
        raise RuntimeError(
            f"{text_boundary_key} was saved as "
            f"{state[text_boundary_key].dtype}"
        )
    if state[image_boundary_key].item() != image_count:
        raise RuntimeError("Saved image count does not match rebuilt count")
    if state[text_boundary_key].item() != text_count:
        raise RuntimeError("Saved text count does not match rebuilt count")


def rebuild_anchors(args):
    if args.task_id < 0:
        raise ValueError("task-id must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / "non_lora_trainables.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing non-LoRA checkpoint: {checkpoint_path}"
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    clip_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.clip_model,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = (
            conversation_lib.conv_templates[args.version]
        )
    else:
        conversation_lib.default_conversation = (
            conversation_lib.conv_templates["vicuna_v1"]
        )

    tower_args = SimpleNamespace(
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        mm_text_select_layer=-1,
    )
    vision_tower = CLIPVisionTower(args.clip_model, tower_args)
    text_tower = CLIPTextTower(args.clip_model, tower_args)
    device = torch.device(args.device)
    tower_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    vision_tower.to(device=device, dtype=tower_dtype).eval()
    text_tower.to(device=device, dtype=tower_dtype).eval()

    data_args = DataArguments(
        data_path=args.data_path,
        lazy_preprocess=True,
        is_multimodal=True,
        image_folder=args.image_folder,
        image_aspect_ratio="pad",
    )
    data_args.mm_use_im_start_end = False
    data_args.image_processor = vision_tower.image_processor
    dataset = LazySupervisedDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    image_sum = None
    text_sum = None
    image_count = 0
    text_count = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Task {args.task_id + 1} anchors"):
            if "images" not in batch:
                raise ValueError("Anchor rebuilding requires image examples")
            images = batch["images"].to(
                device=device,
                dtype=tower_dtype,
                non_blocking=True,
            )
            image_features, _ = vision_tower(images)
            image_sum, image_count = accumulate_rows(
                image_features,
                image_sum,
                image_count,
            )

            clip_prompts = decode_clip_prompts(batch["input_ids"], tokenizer)
            clip_inputs = clip_tokenizer(
                clip_prompts,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            text_features = text_tower(clip_inputs)
            text_sum, text_count = accumulate_rows(
                text_features,
                text_sum,
                text_count,
            )

    if image_count != len(dataset) or text_count != len(dataset):
        raise RuntimeError(
            "Incomplete anchor rebuild: "
            f"dataset={len(dataset)}, image_count={image_count}, "
            f"text_count={text_count}"
        )
    image_anchor = (image_sum / image_count).float().unsqueeze(0)
    text_anchor = (text_sum / text_count).float().unsqueeze(0)
    if not torch.isfinite(image_anchor).all():
        raise RuntimeError("Rebuilt image anchor contains non-finite values")
    if not torch.isfinite(text_anchor).all():
        raise RuntimeError("Rebuilt text anchor contains non-finite values")

    state = torch.load(checkpoint_path, map_location="cpu")
    for key in state:
        if any(statistic_name in key for statistic_name in STATISTIC_NAMES):
            state[key] = state[key].float()

    synchronized_tasks = []
    source_checkpoint_path = None
    source_checkpoint_hash = None
    if args.task_id > 0:
        if args.previous_checkpoint_dir is None:
            raise ValueError(
                "previous-checkpoint-dir is required when task-id is greater "
                "than zero"
            )
        source_checkpoint_path = (
            Path(args.previous_checkpoint_dir) / "non_lora_trainables.bin"
        )
        if not source_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing previous checkpoint: {source_checkpoint_path}"
            )
        source_state = torch.load(source_checkpoint_path, map_location="cpu")
        synchronized_tasks = synchronize_previous_statistics(
            state,
            source_state,
            args.task_id,
        )
        source_checkpoint_hash = checkpoint_sha256(source_checkpoint_path)

    image_anchor_key = find_statistic_key(
        state,
        "image_anchors",
        args.task_id,
    )
    text_anchor_key = find_statistic_key(
        state,
        "text_anchors",
        args.task_id,
    )
    image_boundary_key = find_statistic_key(
        state,
        "image_boundary",
        args.task_id,
    )
    text_boundary_key = find_statistic_key(
        state,
        "text_boundary",
        args.task_id,
    )
    old_image_anchor = state[image_anchor_key].float()
    old_text_anchor = state[text_anchor_key].float()
    old_image_count = state[image_boundary_key].float().item()
    old_text_count = state[text_boundary_key].float().item()

    state[image_anchor_key] = image_anchor
    state[text_anchor_key] = text_anchor
    state[image_boundary_key] = torch.tensor(
        [image_count],
        dtype=torch.float32,
    )
    state[text_boundary_key] = torch.tensor(
        [text_count],
        dtype=torch.float32,
    )

    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )
    torch.save(state, temporary_path)
    candidate_state = torch.load(temporary_path, map_location="cpu")
    validate_rebuilt_state(
        candidate_state,
        image_anchor_key,
        text_anchor_key,
        image_boundary_key,
        text_boundary_key,
        image_count,
        text_count,
    )

    backup_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".pre_anchor_rebuild"
    )
    if not backup_path.exists():
        shutil.copy2(checkpoint_path, backup_path)
    os.replace(temporary_path, checkpoint_path)

    image_similarity = torch.nn.functional.cosine_similarity(
        old_image_anchor,
        image_anchor,
    ).item()
    text_similarity = torch.nn.functional.cosine_similarity(
        old_text_anchor,
        text_anchor,
    ).item()
    metadata = {
        "task_id": args.task_id,
        "task_number": args.task_id + 1,
        "data_path": str(Path(args.data_path)),
        "dataset_samples": len(dataset),
        "image_count": image_count,
        "text_count": text_count,
        "old_image_count": old_image_count,
        "old_text_count": old_text_count,
        "old_new_image_anchor_cosine": image_similarity,
        "old_new_text_anchor_cosine": text_similarity,
        "synchronized_previous_task_ids": synchronized_tasks,
        "source_checkpoint_path": (
            str(source_checkpoint_path)
            if source_checkpoint_path is not None
            else None
        ),
        "source_checkpoint_sha256": source_checkpoint_hash,
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "backup_path": str(backup_path),
        "tower_dtype": str(tower_dtype),
        "saved_dtype": str(state[image_anchor_key].dtype),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = (
        checkpoint_dir / f"anchor_rebuild_task{args.task_id + 1}.json"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(metadata, indent=2))


def main():
    rebuild_anchors(parse_args())


if __name__ == "__main__":
    main()
