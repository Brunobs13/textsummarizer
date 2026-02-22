from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_from_disk
import os

from src.textSummarizer.entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Keep local Macs stable: force CPU by default for this project pipeline.
        device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_pegasus)

        dataset_samsum_pt = load_from_disk(str(self.config.data_path))

        # Local smoke-test settings to avoid MPS OOM / long runs.
        train_ds = dataset_samsum_pt["train"].select(range(min(64, len(dataset_samsum_pt["train"]))))
        eval_ds = dataset_samsum_pt["validation"].select(range(min(16, len(dataset_samsum_pt["validation"]))))

        training_kwargs = dict(
            output_dir=str(self.config.root_dir),
            num_train_epochs=1,
            warmup_steps=0,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=5,
            eval_steps=500,
            save_steps=1000000,
            gradient_accumulation_steps=1,
            max_steps=1,
            dataloader_pin_memory=False,
            report_to=[],
            save_strategy="no",  # avoid checkpoint writes on local smoke tests
        )

        # transformers version compatibility
        try:
            trainer_args = TrainingArguments(**training_kwargs, eval_strategy="no", use_cpu=True)
        except TypeError:
            try:
                trainer_args = TrainingArguments(**training_kwargs, eval_strategy="no")
            except TypeError:
                # Older versions may reject save_strategy and/or eval_strategy naming.
                legacy_kwargs = dict(training_kwargs)
                legacy_kwargs.pop("save_strategy", None)
                trainer_args = TrainingArguments(**legacy_kwargs, evaluation_strategy="no")

        # Defensive: if the attribute exists, keep checkpoint saving disabled for local runs.
        if hasattr(trainer_args, "save_strategy"):
            trainer_args.save_strategy = "no"

        try:
            trainer = Trainer(
                model=model_pegasus,
                args=trainer_args,
                processing_class=tokenizer,
                data_collator=seq2seq_data_collator,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
            )
        except TypeError:
            trainer = Trainer(
                model=model_pegasus,
                args=trainer_args,
                tokenizer=tokenizer,
                data_collator=seq2seq_data_collator,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
            )

        trainer.train()

        # Save tokenizer always; skip large model save locally if disk is tight.
        os.makedirs(str(self.config.root_dir), exist_ok=True)
        tokenizer.save_pretrained(os.path.join(str(self.config.root_dir), "tokenizer"))

        try:
            model_pegasus.save_pretrained(os.path.join(str(self.config.root_dir), "pegasus-samsum-model"))
        except Exception:
            # Allow pipeline to continue to evaluation fallback if local save fails.
            pass
