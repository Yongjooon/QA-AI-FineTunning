# QA-AI-FineTunning

Colab-first project structure for fine-tuning a Gemma 4 model to generate QA test scenarios.

## VS Code + Colab

This repository is set up to run the notebooks directly from VS Code with the Google Colab extension.

Recommended flow:

1. Open this folder in VS Code.
2. Open one of the notebook files in `colab/`.
3. Click `Select Kernel` and choose `Colab` -> `Auto Connect`.
4. Optional but recommended when you have local changes not pushed to GitHub:
   use Explorer right-click on the repository folder and run `Upload to Colab`.
5. Run the first setup cell in the notebook.
   - If the repository already exists on the connected Colab server, the notebook uses it.
   - Otherwise it clones the GitHub repository automatically.

Useful VS Code commands:

- `Colab: Mount Google Drive to Server...`
- `Colab: Open Terminal`
- `Upload to Colab` from the Explorer context menu

## What is included

- `colab/01_train_gemma_qa.ipynb`
  Training page for LoRA fine-tuning from a zipped dataset
- `colab/02_test_gemma_qa.ipynb`
  Inference page for generating Playwright-ready JSON and a human-readable scenario summary
- `src/qafinetune/train.py`
  Training entry point with runtime inspection, checkpointing, logging, and resume support
- `src/qafinetune/infer.py`
  Inference entry point for zipped inputs
- `requirements-colab.txt`
  Colab dependency list

## Expected workflow

1. Open `colab/01_train_gemma_qa.ipynb` in Colab.
2. Install dependencies, mount Google Drive, upload your training zip, and run training.
3. Training artifacts are stored in Google Drive so you can resume from the latest checkpoint.
4. After training completes, open `colab/02_test_gemma_qa.ipynb`.
5. Upload an input zip and generate:
   - one Playwright JSON file
   - one text/markdown summary of the scenario

## Supported training data formats inside the zip

The loader accepts `.json`, `.jsonl`, `.csv`, `.parquet`, `.txt`, and `.md`.

For structured records, it looks for common fields such as:

- prompt fields: `prompt`, `question`, `instruction`, `input`, `user_input`, `context`
- response fields: `response`, `answer`, `output`, `completion`, `target`, `assistant_output`
- message fields: `messages`, `conversation`, `chat`

If a record contains both a Playwright JSON payload and a text summary, the loader can combine them into one assistant response automatically.

## Resume behavior

Each training run writes:

- `runtime_profile.json`
- `dataset_profile.json`
- `run_state.json`
- `logs/train.log`
- `logs/trainer_metrics.jsonl`
- `checkpoints/checkpoint-*`
- `final_model/`

If Colab stops mid-run, rerun the training notebook and keep `RESUME_MODE = "auto"`.
The script will look up the latest checkpoint from the saved run state.

## Note about the base model

The notebooks default to `google/gemma-4-E2B-it`, which is the most practical Gemma 4 option for a T4-backed Colab runtime.
Gemma 4 support requires a newer Transformers release than older Gemma notebooks, so `requirements-colab.txt` now pins a higher minimum version.

## Optional monitoring

The training notebook includes an optional Smankusors Colab Monitor cell.
It is disabled by default because it executes a third-party remote script with `exec()`.
