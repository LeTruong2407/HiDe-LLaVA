from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "hide-llava-ucit-consensus-h100"
GPU_TYPE = "H100"
REPO_ROOT = Path("/root/HiDe-LLaVA")
ASSET_MOUNT = REPO_ROOT / "hide-llava-assets"
OUTPUT_MOUNT = REPO_ROOT / "outputs"

assets_volume = modal.Volume.from_name("hide-llava-assets", create_if_missing=True)
outputs_volume = modal.Volume.from_name("hide-llava-outputs", create_if_missing=True)

TRAIN_PROFILES = {
    "safe": {
        "train_batch": 4,
        "grad_accum": 8,
        "model_max_length": 1024,
        "sample_limit": 128,
        "samples_per_forward": 4,
    },
    "balanced": {
        "train_batch": 8,
        "grad_accum": 4,
        "model_max_length": 1536,
        "sample_limit": 256,
        "samples_per_forward": 8,
    },
    "aggressive": {
        "train_batch": 12,
        "grad_accum": 3,
        "model_max_length": 2048,
        "sample_limit": 256,
        "samples_per_forward": 8,
    },
}
QUANT_MODES = {"bf16", "16bit"}
EVAL_QUANT_MODES = {"fp16", "4bit"}

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "build-essential",
        "curl",
        "git",
        "libgl1",
        "libglib2.0-0",
    )
    .add_local_file(
        "requirements.txt",
        remote_path="/tmp/hide-llava-requirements.txt",
        copy=True,
    )
    .pip_install(
        "pip==24.2",
        "setuptools<70",
        "wheel",
    )
    .run_commands(
        "python -m pip install -r /tmp/hide-llava-requirements.txt",
        (
            "python -m pip install --force-reinstall "
            "torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 "
            "--index-url https://download.pytorch.org/whl/cu121"
        ),
        "python -m pip install --force-reinstall 'numpy==1.23.5'",
        "python -m pip uninstall -y bitsandbytes || true",
    )
    .add_local_dir(
        ".",
        remote_path=str(REPO_ROOT),
        copy=True,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            "hide-llava-assets",
            "outputs",
            "results",
            "wandb",
        ],
    )
    .workdir(str(REPO_ROOT))
    .run_commands(
        "python -m pip install -e .",
    )
)

app = modal.App(APP_NAME, image=image)


def run_checked(command: list[str], env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, cwd=REPO_ROOT, env=merged_env, check=True)


@app.function(
    timeout=8 * 60 * 60,
    volumes={
        str(ASSET_MOUNT): assets_volume,
        str(OUTPUT_MOUNT): outputs_volume,
    },
)
def prepare_assets() -> None:
    env = {
        "HIDE_ASSETS_ROOT": str(ASSET_MOUNT),
    }
    run_checked(["bash", "scripts/download/setup_imagenet_r_assets.sh"], env=env)
    assets_volume.commit()


@app.function(
    timeout=8 * 60 * 60,
    volumes={
        str(ASSET_MOUNT): assets_volume,
    },
)
def prepare_arxivqa_assets() -> None:
    image_dir = ASSET_MOUNT / "datasets" / "ArxivQA" / "images"
    if image_dir.is_dir() and any(image_dir.iterdir()):
        print(f"ArxivQA images already exist at {image_dir}")
        return

    env = {
        "HIDE_ASSETS_ROOT": str(ASSET_MOUNT),
    }
    run_checked(["bash", "scripts/download/download_arxivqa.sh"], env=env)
    assets_volume.commit()


@app.function(
    gpu=GPU_TYPE,
    timeout=24 * 60 * 60,
    volumes={
        str(ASSET_MOUNT): assets_volume,
        str(OUTPUT_MOUNT): outputs_volume,
    },
)
def check_assets() -> None:
    run_checked(["bash", "scripts/download/check_assets.sh"])


@app.function(
    gpu=GPU_TYPE,
    timeout=24 * 60 * 60,
    volumes={
        str(ASSET_MOUNT): assets_volume,
        str(OUTPUT_MOUNT): outputs_volume,
    },
)
def train_ucit_task(
    task_number: int = 1,
    max_steps: int | None = None,
    profile: str = "balanced",
    quant_mode: str = "4bit",
    train_batch: int | None = None,
    grad_accum: int | None = None,
    model_max_length: int | None = None,
    logging_steps: int | None = None,
    sample_limit: int | None = None,
    samples_per_forward: int | None = None,
    save_strategy: str | None = None,
    save_steps: int | None = None,
    save_total_limit: int | None = None,
    resume_from_checkpoint: str | None = None,
) -> None:
    if task_number not in range(1, 7):
        raise ValueError("task_number must be between 1 and 6")
    if profile not in TRAIN_PROFILES:
        raise ValueError(
            f"profile must be one of {sorted(TRAIN_PROFILES)}"
        )
    if quant_mode not in QUANT_MODES:
        raise ValueError(
            f"quant_mode must be one of {sorted(QUANT_MODES)}"
        )
    settings = TRAIN_PROFILES[profile].copy()
    overrides = {
        "train_batch": train_batch,
        "grad_accum": grad_accum,
        "model_max_length": model_max_length,
        "sample_limit": sample_limit,
        "samples_per_forward": samples_per_forward,
    }
    settings.update({
        key: value for key, value in overrides.items()
        if value is not None
    })
    env = {
        "HIDE_ASSETS_ROOT": str(ASSET_MOUNT),
        "UCIT_OUTPUT_ROOT": str(OUTPUT_MOUNT / "ucit_consensus_modal_a100"),
        "TRAIN_PER_DEVICE_BATCH": str(settings["train_batch"]),
        "TRAIN_GRAD_ACCUM_STEPS": str(settings["grad_accum"]),
        "TRAIN_MODEL_MAX_LENGTH": str(settings["model_max_length"]),
        "CONSENSUS_SAMPLE_LIMIT": str(settings["sample_limit"]),
        "CONSENSUS_SAMPLES_PER_FORWARD": str(settings["samples_per_forward"]),
        "CONSENSUS_QUANT_MODE": quant_mode,
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    }
    if logging_steps is not None:
        env["TRAIN_LOGGING_STEPS"] = str(logging_steps)
    if save_strategy is not None:
        env["TRAIN_SAVE_STRATEGY"] = save_strategy
    if save_steps is not None:
        env["TRAIN_SAVE_STEPS"] = str(save_steps)
    if save_total_limit is not None:
        env["TRAIN_SAVE_TOTAL_LIMIT"] = str(save_total_limit)
    if resume_from_checkpoint is not None:
        env["TRAIN_RESUME_FROM_CHECKPOINT"] = resume_from_checkpoint
    if max_steps is not None:
        smoke_output = (
            OUTPUT_MOUNT
            / "smoke"
            / f"Task{task_number}_llava_lora_ours"
        )
        env["EXTRA_TRAIN_ARGS"] = (
            f"--max_steps {max_steps} "
            f"--save_strategy no "
            f"--output_dir {smoke_output}"
        )

    try:
        run_checked(
            [
                "bash",
                "scripts/HiDe/Train_UCIT/run_consensus_task_modal_a100.sh",
                str(task_number),
            ],
            env=env,
        )
    finally:
        outputs_volume.commit()


EVAL_DATASETS = {
    "imagenet-r": {
        "script": "scripts/HiDe/Eval_UCIT/eval_imagenet.sh",
        "result_dir": "ImageNet-R",
    },
    "arxivqa": {
        "script": "scripts/HiDe/Eval_UCIT/eval_arxivqa.sh",
        "result_dir": "ArxivQA",
    },
}


@app.function(
    gpu=GPU_TYPE,
    timeout=12 * 60 * 60,
    volumes={
        str(ASSET_MOUNT): assets_volume,
        str(OUTPUT_MOUNT): outputs_volume,
    },
)
def eval_ucit_task(
    model_task_number: int = 1,
    dataset: str = "imagenet-r",
    stage: str = "consensus-modal-a100-task1",
    max_samples: int | None = None,
    quant_mode: str = "fp16",
    force_expert: int | None = None,
    isolate_expert: bool = False,
    adaptive_all_layer_routing: bool = False,
) -> None:
    if model_task_number not in range(1, 7):
        raise ValueError("model_task_number must be between 1 and 6")
    if dataset not in EVAL_DATASETS:
        raise ValueError(
            f"dataset must be one of {sorted(EVAL_DATASETS)}"
        )
    if (
        force_expert is not None
        and force_expert not in range(model_task_number)
    ):
        raise ValueError(
            "force_expert must reference an expert learned by the checkpoint"
        )
    if isolate_expert and force_expert is None:
        raise ValueError("isolate_expert requires force_expert")
    if quant_mode not in EVAL_QUANT_MODES:
        raise ValueError(
            f"eval quant_mode must be one of {sorted(EVAL_QUANT_MODES)}"
        )
    dataset_config = EVAL_DATASETS[dataset]
    model_path = (
        OUTPUT_MOUNT
        / "ucit_consensus_modal_a100"
        / f"Task{model_task_number}_llava_lora_ours"
    )
    result_dir = (
        OUTPUT_MOUNT
        / "results"
        / "UCIT"
        / "each_dataset"
        / dataset_config["result_dir"]
    )
    required_checkpoint_files = [
        "adapter_config.json",
        "adapter_model.bin",
        "non_lora_trainables.bin",
        "consensus_subspaces.pt",
        "consensus_summary.json",
    ]
    missing = [
        filename for filename in required_checkpoint_files
        if not (model_path / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint is not ready at {model_path}. Missing: {missing}"
        )
    env = {
        "HIDE_ASSETS_ROOT": str(ASSET_MOUNT),
        "RESULT_DIR": str(result_dir),
        "EVAL_GPUS": "0",
        "EVAL_CHUNKS": "1",
        "EVAL_QUANT_ARGS": "--load-4bit" if quant_mode == "4bit" else "",
    }
    if max_samples is not None:
        env["EVAL_MAX_SAMPLES"] = str(max_samples)
    if force_expert is not None:
        env["EVAL_FORCE_EXPERT"] = str(force_expert)
    if isolate_expert:
        env["EVAL_ISOLATE_EXPERT"] = "1"
    if adaptive_all_layer_routing:
        env["EVAL_ADAPTIVE_ALL_LAYER_ROUTING"] = "1"

    run_checked(
        [
            "bash",
            dataset_config["script"],
            stage,
            str(model_path),
            "0",
        ],
        env=env,
    )
    outputs_volume.commit()


@app.local_entrypoint()
def main(
    action: str = "train",
    task_number: int = 1,
    max_steps: int | None = None,
    profile: str = "balanced",
    quant_mode: str = "4bit",
    train_batch: int | None = None,
    grad_accum: int | None = None,
    model_max_length: int | None = None,
    logging_steps: int | None = None,
    sample_limit: int | None = None,
    samples_per_forward: int | None = None,
    save_strategy: str | None = None,
    save_steps: int | None = None,
    save_total_limit: int | None = None,
    resume_from_checkpoint: str | None = None,
    eval_stage: str = "consensus-modal-a100-task1",
    eval_model_task_number: int = 1,
    eval_dataset: str = "imagenet-r",
    eval_max_samples: int | None = None,
    eval_quant_mode: str = "fp16",
    eval_force_expert: int | None = None,
    eval_isolate_expert: bool = False,
    eval_adaptive_all_layer_routing: bool = False,
) -> None:
    if action == "check-assets":
        check_assets.remote()
    elif action == "prepare-assets":
        prepare_assets.remote()
    elif action == "prepare-arxivqa":
        prepare_arxivqa_assets.remote()
    elif action == "train":
        train_ucit_task.spawn(
            task_number=task_number,
            max_steps=max_steps,
            profile=profile,
            quant_mode=quant_mode,
            train_batch=train_batch,
            grad_accum=grad_accum,
            model_max_length=model_max_length,
            logging_steps=logging_steps,
            sample_limit=sample_limit,
            samples_per_forward=samples_per_forward,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    elif action == "eval":
        eval_ucit_task.spawn(
            model_task_number=eval_model_task_number,
            dataset=eval_dataset,
            stage=eval_stage,
            max_samples=eval_max_samples,
            quant_mode=eval_quant_mode,
            force_expert=eval_force_expert,
            isolate_expert=eval_isolate_expert,
            adaptive_all_layer_routing=(
                eval_adaptive_all_layer_routing
            ),
        )
    else:
        raise ValueError(
            "action must be 'prepare-assets', 'prepare-arxivqa', "
            "'check-assets', 'train', or 'eval'"
        )
