from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk

import torch
import pandas as pd
from tqdm import tqdm
import evaluate

from src.textSummarizer.entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size=2,
        device="cpu",
        column_text="dialogue",
        column_summary="summary",
    ):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128,
            )

            decoded_summaries = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for s in summaries
            ]
            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        return metric.compute()

    def _load_model_and_tokenizer(self, device):
        model_path = Path(self.config.model_path)
        tokenizer_path = Path(self.config.tokenizer_path)

        if model_path.exists() and tokenizer_path.exists():
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
                model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), local_files_only=True).to(device)
                return model_pegasus, tokenizer, "local"
            except Exception:
                pass

        base_model = "google/pegasus-cnn_dailymail"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
        return model_pegasus, tokenizer, "hf_base"

    def evaluate(self):
        # Keep local Macs stable.
        device = "cpu"

        tokenizer = None
        model_pegasus = None
        dataset_samsum_pt = load_from_disk(str(self.config.data_path))

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = evaluate.load("rouge")

        model_pegasus, tokenizer, _ = self._load_model_and_tokenizer(device=device)

        test_sample = dataset_samsum_pt["test"].select(range(min(10, len(dataset_samsum_pt["test"]))))
        score = self.calculate_metric_on_test_ds(
            test_sample,
            rouge_metric,
            model_pegasus,
            tokenizer,
            batch_size=2,
            device=device,
            column_text="dialogue",
            column_summary="summary",
        )

        rouge_dict = {rn: score[rn] for rn in rouge_names}
        metric_path = Path(self.config.metric_file_name)
        metric_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(rouge_dict, index=["pegasus"])
        df.to_csv(metric_path, index=False)
