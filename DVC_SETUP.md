# DVC + GCS Setup

This guide stores `hide-llava-assets/` in Google Cloud Storage through DVC, while Git keeps only metadata such as `.dvc` files and config.

## Why this setup

- keeps model weights and datasets out of Git
- makes local machine and Kaggle/Linux pulls consistent
- lets us version `hide-llava-assets/` as one reproducible bundle

## 1. Install DVC with GCS support

Inside your active environment:

```bash
python -m pip install -r requirements.dvc.txt
```

This installs `dvc` with Google Cloud Storage support via `dvc[gs]`.

## 2. Create or choose a GCS bucket

Pick a bucket and optional prefix, for example:

```bash
gs://your-hide-llava-bucket/hide-llava-assets
```

## 3. Initialize DVC and configure the remote

From the repo root:

```bash
bash scripts/dvc/setup_gcs_remote.sh gs://dvc_example_project/dvc_example_project

```

This will:

- run `dvc init` if needed
- create a default DVC remote named `gcs`
- store the remote URL in `.dvc/config`

## 4. Authenticate to GCS

Recommended options:

### Option A: Application Default Credentials

```bash
gcloud auth application-default login
```

### Option B: Service account key

```bash
dvc remote modify --local gcs credentialpath /absolute/path/to/service-account.json
```

`config.local` stays local-only and should not be committed.

## 5. Track and push `hide-llava-assets`

```bash
bash scripts/dvc/push_assets.sh
```

This runs:

- `dvc add hide-llava-assets`
- `git add hide-llava-assets.dvc .gitignore`
- `dvc push`

After that, commit the metadata:

```bash
git add .dvc/config hide-llava-assets.dvc .gitignore
git commit -m "Track hide-llava-assets with DVC"
```

Do not commit `.dvc/config.local`.

## 6. Pull assets on another machine

On Kaggle/Linux or any new machine:

```bash
python -m pip install -r requirements.dvc.txt
bash scripts/dvc/pull_assets.sh
```

If credentials are not already available in the environment, configure them first.

## 7. Push trained checkpoints

After a task finishes training, push the task output directory to the same GCS
DVC remote:

```bash
bash scripts/dvc/push_checkpoint.sh outputs/ucit_consensus/Task1_llava_lora_ours
```

For a specific remote name:

```bash
bash scripts/dvc/push_checkpoint.sh outputs/ucit_consensus/Task1_llava_lora_ours gcs
```

The script runs:

- `dvc add CHECKPOINT_DIR`
- `git add CHECKPOINT_DIR.dvc .gitignore`
- `dvc push`

For the consensus method, the checkpoint should include:

```text
adapter_config.json
adapter_model.bin
non_lora_trainables.bin
consensus_subspaces.pt
consensus_summary.json
```

Commit the generated `.dvc` metadata after pushing, for example:

```bash
git add outputs/ucit_consensus/Task1_llava_lora_ours.dvc .gitignore
git commit -m "track task1 consensus checkpoint with DVC"
```

To pull the checkpoint on another machine, commit/pull the `.dvc` file first,
then run:

```bash
dvc pull outputs/ucit_consensus/Task1_llava_lora_ours.dvc
```

## 8. Suggested workflow

### Local machine

```bash
conda activate hide-llava
python -m pip install -r requirements.dvc.txt
bash scripts/dvc/setup_gcs_remote.sh gs://your-hide-llava-bucket/hide-llava-assets
bash scripts/dvc/push_assets.sh
```

### Kaggle/Linux

```bash
python -m pip install -r requirements.dvc.txt
gcloud auth application-default login
bash scripts/dvc/pull_assets.sh
```

Then run the project checks or training commands as usual.

### After each training task

```bash
bash scripts/dvc/push_checkpoint.sh outputs/ucit_consensus/Task1_llava_lora_ours
git commit -m "track task1 consensus checkpoint with DVC"
```

Repeat with `Task2_llava_lora_ours`, `Task3_llava_lora_ours`, and so on.

## 9. What gets committed

Commit:

- `.dvc/config`
- `hide-llava-assets.dvc`
- `outputs/.../Task*_llava_lora_ours.dvc`
- `.gitignore`

Keep local-only:

- `.dvc/config.local`
- actual contents of `hide-llava-assets/`
- actual contents of `outputs/.../Task*_llava_lora_ours/`

## 10. Notes

- As of July 23, 2026, the standard DVC flow for GCS is to install `dvc[gs]`, add a `gs://...` remote, and configure auth either through `gcloud auth application-default login` or a service-account credential path in local config, based on DVC’s official guidance.
- `hide-llava-assets/` can be large, so the first `dvc push` may take a while.
- If you want finer granularity later, we can split it into multiple tracked targets such as `hide-llava-assets/models`, `hide-llava-assets/datasets`, and `hide-llava-assets/instructions`.
