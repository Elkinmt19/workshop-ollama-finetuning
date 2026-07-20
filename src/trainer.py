"""LoRA fine-tuning trainer for TinyLlama, using transformers + peft."""

import logging
from pathlib import Path
from typing import Optional

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data_loader import DataLoader as ExampleLoader
from src.data_loader import TrainingExample

logger = logging.getLogger(__name__)


class ChatDataset(Dataset):
    """Tokenizes chat-formatted examples with prompt tokens masked out of the loss."""

    def __init__(self, examples: list[TrainingExample], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = [self._encode(ex) for ex in examples]

    def _encode(self, example: TrainingExample) -> dict:
        messages = example.to_chat_messages()
        prompt_text = self.tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize the rendered text directly rather than trusting apply_chat_template's
        # tokenize=True return type, which varies across transformers/tokenizers versions
        # (some return plain int lists, others return tokenizers.Encoding objects).
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        full_ids = full_ids[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))

        labels = list(full_ids)
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class PadCollator:
    """Pads variable-length input_ids/attention_mask/labels to the batch max length."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(row["input_ids"]) for row in batch)

        input_ids, attention_mask, labels = [], [], []
        for row in batch:
            pad_len = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(row["attention_mask"] + [0] * pad_len)
            labels.append(row["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class OllamaTrainer:
    """Orchestrates LoRA fine-tuning of a Hugging Face model for later Ollama serving."""

    def __init__(self, config_path: Path):
        """Initialize trainer with config file."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Resolve relative paths from project root (config lives in <root>/config/)
        self.project_root = config_path.parent.parent
        self.model_name = self.config["model"]["base_model_hf"]
        self.output_dir = Path(self.config["output"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_data_path(self, relative_path: str) -> Path:
        """Resolve a data path from config relative to project root."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return self.project_root / p

    def validate_config(self) -> bool:
        """Validate training configuration."""
        required_keys = ["model", "training", "data", "output"]
        for key in required_keys:
            if key not in self.config:
                logger.error(f"Missing required config key: {key}")
                return False

        for data_type in ["train_file", "eval_file"]:
            path = self._resolve_data_path(self.config["data"][data_type])
            if not path.exists():
                logger.warning(f"{data_type} not found: {path}")

        return True

    def _load_examples(self, key: str) -> list[TrainingExample]:
        path = self._resolve_data_path(self.config["data"][key])
        return ExampleLoader.load_dataset(path)

    def train(self) -> dict:
        """Run LoRA fine-tuning and save both the adapter and the merged model."""
        if not self.validate_config():
            raise ValueError("Invalid configuration")

        training_cfg = self.config["training"]
        output_cfg = self.config["output"]

        logger.info(f"Starting fine-tuning with model: {self.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        if training_cfg.get("use_lora", True):
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=training_cfg.get("lora_rank", 16),
                lora_alpha=training_cfg.get("lora_alpha", 32),
                lora_dropout=training_cfg.get("lora_dropout", 0.05),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        max_length = training_cfg.get("max_length", 512)
        train_dataset = ChatDataset(self._load_examples("train_file"), tokenizer, max_length)
        eval_examples = self._load_examples("eval_file")
        eval_dataset = (
            ChatDataset(eval_examples, tokenizer, max_length) if eval_examples else None
        )

        checkpoint_dir = self.output_dir / "checkpoint"
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=training_cfg["epochs"],
            per_device_train_batch_size=training_cfg["batch_size"],
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
            learning_rate=float(training_cfg["learning_rate"]),
            warmup_ratio=float(training_cfg.get("warmup_ratio", 0.1)),
            save_steps=output_cfg.get("save_steps", 50),
            save_total_limit=output_cfg.get("save_total_limit", 3),
            eval_steps=output_cfg.get("eval_steps", 25),
            eval_strategy="steps" if eval_dataset else "no",
            logging_steps=output_cfg.get("logging_steps", 5),
            load_best_model_at_end=output_cfg.get("load_best_model_at_end", True) if eval_dataset else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=(device == "cuda"),
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=PadCollator(tokenizer.pad_token_id),
        )

        trainer.train()

        adapter_dir = self.output_dir / "adapter"
        merged_dir = self.output_dir / "merged"

        if training_cfg.get("use_lora", True):
            model.save_pretrained(str(adapter_dir))
            merged_model = model.merge_and_unload()
        else:
            merged_model = model

        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

        log_history = trainer.state.log_history
        train_losses = [entry["loss"] for entry in log_history if "loss" in entry]

        result = {
            "model": self.model_name,
            "epochs": training_cfg["epochs"],
            "status": "completed",
            "final_loss": train_losses[-1] if train_losses else None,
            "log_history": log_history,
            "adapter_dir": str(adapter_dir) if training_cfg.get("use_lora", True) else None,
            "merged_dir": str(merged_dir),
        }

        logger.info(f"Fine-tuning complete: {result}")
        return result

    def get_config_summary(self) -> str:
        """Return human-readable config summary."""
        training = self.config["training"]
        return f"""
Fine-tuning Configuration:
  Model: {self.model_name}
  Epochs: {training["epochs"]}
  Batch Size: {training["batch_size"]}
  Learning Rate: {training["learning_rate"]}
  LoRA: {training.get("use_lora", False)}
"""
