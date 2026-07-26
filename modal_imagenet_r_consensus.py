from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "hide-llava-imagenet-r-consensus"
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
QUANT_MODES = {"4bit", "bf16", "16bit"}
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
    .pip_install(
        "pip==24.2",
        "setuptools<70",
        "wheel",
    )
    .run_commands(
        "python -m pip install -r requirements.txt",
        "python -m pip install bitsandbytes==0.41.0 triton==2.0.0",
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
    gpu="A100-40GB",
    timeout=24 * 60 * 60,
    volumes={
        str(ASSET_MOUNT): assets_volume,
        str(OUTPUT_MOUNT): outputs_volume,
    },
)
def check_assets() -> None:
    run_checked(["bash", "scripts/download/check_assets.sh"])


@app.function(
    gpu="A100-40GB",
    timeout=24 * 60 * 60,
    volumes={
        str(ASSET_MOUNT): assets_volume,
        str(OUTPUT_MOUNT): outputs_volume,
    },
)
def train_task1(
    max_steps: int | None = None,
    profile: str = "balanced",
    quant_mode: str = "4bit",
    train_batch: int | None = None,
    grad_accum: int | None = None,
    model_max_length: int | None = None,
    logging_steps: int | None = None,
    sample_limit: int | None = None,
    samples_per_forward: int | None = None,
) -> None:
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
    }
    if logging_steps is not None:
        env["TRAIN_LOGGING_STEPS"] = str(logging_steps)
    if max_steps is not None:
        env["EXTRA_TRAIN_ARGS"] = f"--max_steps {max_steps}"

    try:
        run_checked(
            ["bash", "scripts/HiDe/Train_UCIT/run_task1_consensus_modal_a100.sh"],
            env=env,
        )
    finally:
        outputs_volume.commit()


@app.function(
    gpu="A100-40GB",
    timeout=12 * 60 * 60,
    volumes={
        str(ASSET_MOUNT): assets_volume,
        str(OUTPUT_MOUNT): outputs_volume,
    },
)
def eval_task1(
    stage: str = "consensus-modal-a100-task1",
    max_samples: int | None = None,
    quant_mode: str = "fp16",
) -> None:
    if quant_mode not in EVAL_QUANT_MODES:
        raise ValueError(
            f"eval quant_mode must be one of {sorted(EVAL_QUANT_MODES)}"
        )
    model_path = OUTPUT_MOUNT / "ucit_consensus_modal_a100" / "Task1_llava_lora_ours"
    result_dir = OUTPUT_MOUNT / "results" / "UCIT" / "each_dataset" / "ImageNet-R"
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

    run_checked(
        [
            "bash",
            "scripts/HiDe/Eval_UCIT/eval_imagenet.sh",
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
    max_steps: int | None = None,
    profile: str = "balanced",
    quant_mode: str = "4bit",
    train_batch: int | None = None,
    grad_accum: int | None = None,
    model_max_length: int | None = None,
    logging_steps: int | None = None,
    sample_limit: int | None = None,
    samples_per_forward: int | None = None,
    eval_stage: str = "consensus-modal-a100-task1",
    eval_max_samples: int | None = None,
    eval_quant_mode: str = "fp16",
) -> None:
    if action == "check-assets":
        check_assets.remote()
    elif action == "prepare-assets":
        prepare_assets.remote()
    elif action == "train":
        train_task1.remote(
            max_steps=max_steps,
            profile=profile,
            quant_mode=quant_mode,
            train_batch=train_batch,
            grad_accum=grad_accum,
            model_max_length=model_max_length,
            logging_steps=logging_steps,
            sample_limit=sample_limit,
            samples_per_forward=samples_per_forward,
        )
    elif action == "eval":
        eval_task1.remote(
            stage=eval_stage,
            max_samples=eval_max_samples,
            quant_mode=eval_quant_mode,
        )
    else:
        raise ValueError("action must be 'prepare-assets', 'check-assets', 'train', or 'eval'")
