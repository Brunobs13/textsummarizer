from pathlib import Path

# Resolve project root from this file location: src/textSummarizer/constants/__init__.py
ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"
