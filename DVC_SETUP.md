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
bash scripts/dvc/setup_gcs_remote.sh gs://your-hide-llava-bucket/hide-llava-assets
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

## 7. Suggested workflow

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

## 8. What gets committed

Commit:

- `.dvc/config`
- `hide-llava-assets.dvc`
- `.gitignore`

Keep local-only:

- `.dvc/config.local`
- actual contents of `hide-llava-assets/`

## 9. Notes

- As of July 23, 2026, the standard DVC flow for GCS is to install `dvc[gs]`, add a `gs://...` remote, and configure auth either through `gcloud auth application-default login` or a service-account credential path in local config, based on DVC’s official guidance.
- `hide-llava-assets/` can be large, so the first `dvc push` may take a while.
- If you want finer granularity later, we can split it into multiple tracked targets such as `hide-llava-assets/models`, `hide-llava-assets/datasets`, and `hide-llava-assets/instructions`.
