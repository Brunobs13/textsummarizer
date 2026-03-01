# Text Summarizer NLP Pipeline (Pegasus + FastAPI)

Production-style NLP project for abstractive text summarization, built with Hugging Face Transformers and exposed through a FastAPI service.

## Project Summary
This repository implements a complete summarization workflow:
- dataset ingestion
- tokenization and transformation
- model training (local smoke-test mode)
- model evaluation with ROUGE metrics
- API endpoint for inference

The project uses the **Google Pegasus model** as its core summarization backbone.

## Stack (Explicit)
- Python
- FastAPI + Uvicorn
- Hugging Face `transformers`
- Hugging Face `datasets`
- PyTorch
- `evaluate`, `rouge_score`, `sacrebleu`
- Jinja2
- YAML-driven config (`python-box`, `PyYAML`)

## Model Details
- Base model checkpoint: `google/pegasus-cnn_dailymail`
- Task: abstractive summarization
- Dataset source: SAMSum (downloaded from configured URL)
- Evaluation output: ROUGE metrics in CSV (`artifacts/model_evaluation/metrics.csv`)

## Pipeline Workflow
The training pipeline in `main.py` runs these stages in sequence:
1. **Data Ingestion**
   - downloads and extracts summarization dataset
2. **Data Transformation**
   - tokenizes dialogue/summary pairs for seq2seq training
3. **Model Trainer**
   - trains Pegasus in local-safe smoke-test mode
4. **Model Evaluation**
   - computes ROUGE metrics on test subset

## API Workflow
`app.py` exposes:
- `GET /` -> redirects to API docs
- `GET /train` -> runs `python main.py`
- `POST /predict` -> returns generated summary from input text

## Project Structure
```text
textsummarizer/
├── app.py
├── main.py
├── config/
│   └── config.yaml
├── params.yaml
├── src/textSummarizer/
│   ├── components/
│   ├── config/
│   ├── constants/
│   ├── entity/
│   ├── pipeline/
│   ├── logging/
│   └── utils/
├── research/
├── Dockerfile
└── README.md
```

## Run Locally
### 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Train pipeline
```bash
python main.py
```

### 3) Run API
```bash
python app.py
```
Open: `http://localhost:8080/docs`

### 4) Predict (example)
```bash
curl -X POST "http://localhost:8080/predict?text=Your%20long%20dialogue%20here"
```

## Configuration
Main runtime files:
- `config/config.yaml` -> paths, model checkpoint, data URLs
- `params.yaml` -> training argument defaults
- `src/textSummarizer/constants/__init__.py` -> base paths

## Notes
- The trainer is configured for local smoke testing (CPU-safe, minimal steps) to keep runs stable on local machines.
- Evaluation includes fallback behavior to load the base Pegasus model if local fine-tuned artifacts are unavailable.
- `Dockerfile` exists as repository scaffold and can be finalized for deployment runtime packaging.

## What This Project Demonstrates
- Practical NLP MLOps pipeline design
- Config-driven stage orchestration
- Hugging Face training/evaluation integration
- FastAPI-based model serving
- Clean separation between components and execution pipelines
