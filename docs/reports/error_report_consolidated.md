# Consolidated Error Report (Stages 1-4)

## Scope
This report consolidates the main issues encountered while running the end-to-end Text Summarizer project locally (macOS) and in notebooks, plus the applied fixes/workarounds.

## Stage 1 - Data Ingestion

### Issue: File path / environment confusion (multiple project folders)
Cause:
- Work was done across two different project paths (`.../End-to-End MLOps/...` and `.../End-to-End MLOps Bootcamp/...`).
Fix:
- Standardized commands and execution on the active repo path.
- Confirmed generated artifacts in `artifacts/data_ingestion/` before proceeding.

## Stage 2 - Data Transformation (`research/2_data_transformation.ipynb`, component)

### 1. `AttributeError: PegasusTokenizer has no attribute as_target_tokenizer`
Cause:
- Deprecated API usage with newer `transformers` versions.
Fix:
- Replaced target tokenization with `text_target=...`.
- Patched notebook cells and modular component:
  - `research/2_data_transformation.ipynb`
  - `research/textsummarizer.ipynb`
  - `src/textSummarizer/components/data_transformation.py`

### 2. `FileNotFoundError: config/config.yaml`
Cause:
- Relative paths resolved from the wrong notebook working directory.
Fix:
- Added project-root path resolution in constants and notebook setup/config cells.
- Updated `src/textSummarizer/constants/__init__.py` to use absolute project-root-based paths.

### 3. `PermissionError` / `Read-only file system: 'artifacts'`
Cause:
- Notebook kernel running from an unexpected working directory; relative `artifacts/` pointed to a read-only location.
Fix:
- Added a robust notebook setup cell to normalize `cwd` to the project root.
- Added path validation prints.

### 4. `FileNotFoundError: artifacts/data_ingestion/samsum_dataset not found`
Cause:
- Relative `data_path` being resolved from the wrong directory / old execution cells.
Fix:
- Updated execution cells to force absolute paths for `data_path` and `root_dir` before `convert()`.
- Added `exists()` checks before execution.

### 5. Stale Jupyter state (old classes/cells still in memory)
Cause:
- Kernel held previous class definitions despite notebook file being corrected.
Fix:
- Reexecuted corrected class cells.
- Added temporary monkey-patch / hotfix cell for `DataTransformation.convert_examples_to_features` in the notebook during debugging.

## Stage 3 - Model Trainer (`research/3_model_trainer.ipynb`, component, `main.py`)

### 6. HF model loading instability / remote download issues
Cause:
- Local HF Hub behavior and download path instability with modern stack.
Fix:
- Used runtime environment variables in notebook debugging when needed:
  - `HF_HUB_DISABLE_XET=1`
  - `HF_HUB_ENABLE_HF_TRANSFER=0`
- Retried `from_pretrained(...)`.

### 7. `TypeError` in `TrainingArguments` (`evaluation_strategy` unsupported)
Cause:
- Installed `transformers` version expects `eval_strategy`.
Fix:
- Updated notebook and modular trainer compatibility handling to use `eval_strategy` (with fallbacks where useful).

### 8. `TypeError` in `Trainer` (`tokenizer=` unsupported)
Cause:
- Installed `transformers` version expects `processing_class=tokenizer`.
Fix:
- Updated notebook and modular trainer to use `processing_class=tokenizer` with compatibility fallback.

### 9. `RuntimeError: MPS backend out of memory`
Cause:
- macOS MPS memory insufficient for Pegasus training (even during Trainer setup/training).
Fix:
- Forced local smoke tests to CPU.
- Reduced training load drastically:
  - tiny train/validation subsets
  - `max_steps` minimal
  - reduced `gradient_accumulation_steps`
  - local smoke-test mode only

### 10. `safetensors ... No space left on device` during training/checkpoint save
Cause:
- Local disk space exhausted while Trainer checkpointing / model serialization.
Fix:
- Disabled Trainer checkpoint saving for local smoke tests (`save_strategy="no"`).
- Kept final model save guarded with `try/except` so pipeline can continue locally.
- Recommended full training/save on Colab GPU.

### 11. `Kernel crashed` during local training
Cause:
- Resource pressure (MPS OOM + disk pressure) during experiments.
Fix:
- Ultra-light local smoke test config.
- Moved real training recommendation to Google Colab GPU.

### 12. `AttributeError: 'ConfigurationManager' object has no attribute 'params'`
Cause:
- Typo in modular code (`self.paramss` vs `self.params`).
Fix:
- Corrected `ConfigurationManager` to use `self.params` consistently.
- Fixed trainer config field mapping (`eval_steps` source was wrong).

### 13. `ImportError: accelerate>=1.1.0 required`
Cause:
- `Trainer` requires `accelerate` package but it was missing from `.venv`.
Fix:
- Installed `accelerate>=1.1.0` in the project environment.
- Recommendation: add `accelerate` to `requirements.txt`.

## Stage 4 - Model Evaluation (`research/4_model_evaluation.ipynb`, component)

### 14. `SafetensorError` loading local trained model
Cause:
- Local saved model was incomplete/corrupted due to prior disk-space/save failures.
Fix:
- Added fallback logic to use base model `google/pegasus-cnn_dailymail` when local model loading fails.

### 15. `HFValidationError` (local path interpreted as Hugging Face repo id)
Cause:
- Relative local paths like `artifacts/model_trainer/tokenizer` passed to `from_pretrained(...)` and interpreted as HF repo ids when not resolved correctly.
Fix:
- Normalized paths to absolute paths in notebook/component before calling `from_pretrained(...)`.
- Added `local_files_only=True` for local model/tokenizer loading path.

### 16. Local evaluation resource concerns (MPS)
Cause:
- Running evaluation on MPS can trigger memory issues.
Fix:
- Forced local evaluation to CPU and limited test sample to a small subset for smoke test.

## Environment / Tooling Issues

### 17. `.venv` active in prompt but `python` still points to Anaconda
Cause:
- Shell path precedence / mixed environments; active prompt did not guarantee active interpreter.
Fix:
- Verified with `which python` and `sys.executable`.
- Used explicit interpreter path: `./.venv/bin/python`.
- Recreated/fixed environment usage guidance.

## Improvements Applied (Code + Workflow)

- Project-root absolute path constants (`config.yaml`, `params.yaml`).
- Transformers compatibility updates (`text_target`, `eval_strategy`, `processing_class`).
- Local CPU-safe smoke test defaults for `ModelTrainer` and `ModelEvaluation`.
- Trainer save/checkpoint suppression for local low-disk environments.
- Evaluation fallback to base Pegasus when local fine-tuned model is invalid.
- Clear split usage guidance:
  - Train: `train`
  - Validation: `validation`
  - Test: reserved for final evaluation
- Safer handling of private internal reports (`reports/`, `reports_local/`, `notes_private/` ignored).

## Final Local Outcome (Validated)

- `main.py` executed through:
  - Data Ingestion ✅
  - Data Transformation ✅
  - Model Trainer (local smoke test) ✅
  - Model Evaluation (with fallback to base model when local model invalid) ✅
- Metrics file generated at:
  - `artifacts/model_evaluation/metrics.csv`

## Recommended Next Step
- Run full training on Google Colab GPU, save model/tokenizer there, sync artifacts back if needed, then run full evaluation on the fine-tuned model.
