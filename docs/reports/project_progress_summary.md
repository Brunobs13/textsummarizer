# Project Progress Summary - Text Summarizer (End-to-End NLP MLOps)

## Objective
Convert a notebook-based Hugging Face text summarization workflow into a modular production-style project with components, pipelines, and API support.

## Architecture (Defined)
- Data Ingestion
- Data Transformation
- Model Trainer
- Model Evaluation
- Training Pipeline
- Prediction Pipeline
- API Layer

## What Is Implemented / Validated

### 1. Data Ingestion
Status: Implemented and validated
- Downloads dataset zip
- Extracts dataset files
- Produces ingestion artifacts under `artifacts/data_ingestion/`

### 2. Data Transformation
Status: Implemented and validated
- Loads dataset from disk (Hugging Face `DatasetDict`)
- Tokenizes input/output for Pegasus summarization
- Saves transformed dataset to:
  - `artifacts/data_transformation/samsum_dataset`

### 3. Model Trainer (Local)
Status: Pipeline validated via smoke test
- Trainer component updated for local stability and transformers compatibility
- Uses CPU-safe local smoke test mode to avoid MPS OOM / disk issues
- Uses correct split strategy (`train` + `validation`)
- Tokenizer save is attempted; model save is tolerated/fallback-safe if local disk is constrained

### 4. Model Evaluation (Local)
Status: Validated end-to-end
- Loads transformed dataset
- Computes ROUGE on a local smoke-test subset (`test[:10]`)
- Handles invalid/corrupted local model by falling back to base Pegasus model
- Saves metrics CSV under:
  - `artifacts/model_evaluation/metrics.csv`

## Key Codebase Improvements Applied

### Configuration / Paths
- Absolute project-root based constants for `config/config.yaml` and `params.yaml`
- Fixed `ConfigurationManager` params typo and trainer config mapping bugs

### Transformers Compatibility
- Replaced deprecated target tokenizer pattern with `text_target=...`
- Updated `TrainingArguments` usage for newer `transformers` (`eval_strategy`)
- Updated `Trainer` initialization (`processing_class=tokenizer` where needed)

### Local Mac Stability
- CPU-safe execution paths for trainer/evaluation
- Smoke-test subsets and low-step training for local validation
- Checkpoint save suppression during local smoke test to avoid disk exhaustion

### Robust Evaluation
- Local path normalization for model/tokenizer loading
- `local_files_only=True` for local artifacts
- Automatic fallback to `google/pegasus-cnn_dailymail` if local saved model is invalid

## Notebook Progress (Practical)
- `research/2_data_transformation.ipynb`: corrected and validated
- `research/3_model_trainer.ipynb`: corrected for local smoke test and compatibility
- `research/4_model_evaluation.ipynb`: corrected with local fallback and path handling

## Current Local Execution Outcome (`main.py`)
`main.py` runs through the full local pipeline successfully (with local-safe settings):
- Data Ingestion ✅
- Data Transformation ✅
- Model Trainer ✅ (smoke test)
- Model Evaluation ✅ (fallback model if local fine-tuned model invalid)

## Recommended Production/Next Steps
1. Run full training on Google Colab GPU (full dataset, full save of model/tokenizer)
2. Sync trained model artifacts back to project
3. Run full evaluation against the fine-tuned model on `test`
4. Validate/serve through `app.py` (FastAPI)
5. Optional: split local smoke-test behavior vs full-train behavior via config flag/environment variable
