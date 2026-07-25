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
    train_batch: int = 4,
    grad_accum: int = 8,
    model_max_length: int = 1024,
    sample_limit: int = 128,
) -> None:
    env = {
        "HIDE_ASSETS_ROOT": str(ASSET_MOUNT),
        "UCIT_OUTPUT_ROOT": str(OUTPUT_MOUNT / "ucit_consensus_modal_a100"),
        "TRAIN_PER_DEVICE_BATCH": str(train_batch),
        "TRAIN_GRAD_ACCUM_STEPS": str(grad_accum),
        "TRAIN_MODEL_MAX_LENGTH": str(model_max_length),
        "CONSENSUS_SAMPLE_LIMIT": str(sample_limit),
    }
    if max_steps is not None:
        env["EXTRA_TRAIN_ARGS"] = f"--max_steps {max_steps}"

    run_checked(
        ["bash", "scripts/HiDe/Train_UCIT/run_task1_consensus_modal_a100.sh"],
        env=env,
    )
    outputs_volume.commit()


@app.local_entrypoint()
def main(
    action: str = "train",
    max_steps: int | None = None,
    train_batch: int = 4,
    grad_accum: int = 8,
    model_max_length: int = 1024,
    sample_limit: int = 128,
) -> None:
    if action == "check-assets":
        check_assets.remote()
    elif action == "prepare-assets":
        prepare_assets.remote()
    elif action == "train":
        train_task1.remote(
            max_steps=max_steps,
            train_batch=train_batch,
            grad_accum=grad_accum,
            model_max_length=model_max_length,
            sample_limit=sample_limit,
        )
    else:
        raise ValueError("action must be 'prepare-assets', 'check-assets', or 'train'")
