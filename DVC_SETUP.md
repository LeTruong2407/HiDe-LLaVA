# DVC + GCS For Training Artifacts

This guide stores only generated artifacts in Google Cloud Storage through DVC.
An artifact is one file or folder produced by this repo, usually:

- a trained task checkpoint under `outputs/`
- an evaluation result folder under `results/`

Datasets and base models are not managed by DVC in this workflow. Prepare those
with the scripts under `scripts/download/`.

## 1. Install DVC with GCS support

```bash
python3 -m pip install -r requirements.dvc.txt
```

## 2. Configure the GCS remote

Pick a bucket and optional prefix, for example:

```bash
gs://your-hide-llava-bucket/hide-llava-artifacts
```

From the repo root:

```bash
bash scripts/dvc/setup_gcs_remote.sh gs://your-hide-llava-bucket/hide-llava-artifacts
```

This creates a default DVC remote named `gcs`.

## 3. Authenticate to GCS

Application Default Credentials:

```bash
gcloud auth application-default login
```

or a service account key:

```bash
dvc remote modify --local gcs credentialpath /absolute/path/to/service-account.json
```

Do not commit `.dvc/config.local`.

## 4. Push One Artifact

Push a trained checkpoint folder:

```bash
bash scripts/dvc/push_artifact.sh outputs/ucit_consensus/Task1_llava_lora_ours
```

Push an evaluation result folder:

```bash
bash scripts/dvc/push_artifact.sh results/UCIT/each_dataset/ImageNet-R/consensus-task1
```

For a named remote:

```bash
bash scripts/dvc/push_artifact.sh outputs/ucit_consensus/Task1_llava_lora_ours gcs
```

The script runs:

```bash
dvc add --force ARTIFACT_PATH
git add -f ARTIFACT_PATH.dvc .gitignore
dvc push ARTIFACT_PATH.dvc
```

Then commit the DVC metadata:

```bash
git commit -m "track task1 consensus checkpoint with DVC"
```

## 5. Pull One Artifact

First make sure the corresponding `.dvc` file exists locally from Git. Then run:

```bash
bash scripts/dvc/pull_artifact.sh outputs/ucit_consensus/Task1_llava_lora_ours
```

or pass the `.dvc` file directly:

```bash
bash scripts/dvc/pull_artifact.sh outputs/ucit_consensus/Task1_llava_lora_ours.dvc
```

For evaluation output:

```bash
bash scripts/dvc/pull_artifact.sh results/UCIT/each_dataset/ImageNet-R/consensus-task1
```

## 6. What Is Inside Each Artifact

A consensus checkpoint artifact should contain:

```text
adapter_config.json
adapter_model.bin
non_lora_trainables.bin
consensus_subspaces.pt
consensus_summary.json
```

An ImageNet-R eval artifact usually contains:

```text
1_0.jsonl
merge.jsonl
Result.text
```

## 7. What Gets Committed

Commit:

- `.dvc/config`
- `.gitignore`
- `outputs/.../*.dvc`
- `results/.../*.dvc`

Keep local-only:

- `.dvc/config.local`
- actual checkpoint folders under `outputs/`
- actual evaluation result folders under `results/`
- downloaded datasets and base models under `hide-llava-assets/`

## 8. Dataset and Base Model Workflow

Use download scripts for datasets and base models:

```bash
bash scripts/download/setup_all_ucit_assets.sh
bash scripts/download/check_assets.sh
```

DVC pull should not be used to fetch all datasets in this workflow.
