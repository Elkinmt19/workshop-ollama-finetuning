"""Evaluation utilities for fine-tuned models: exact/normalized match, similarity, perplexity."""

import difflib
import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_loader import DataLoader as ExampleLoader
from src.data_loader import TrainingExample

logger = logging.getLogger(__name__)


def _load_model_and_tokenizer(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer, device


def _generate(model, tokenizer, device, example: TrainingExample, max_new_tokens: int = 128) -> str:
    messages = [example.to_chat_messages()[0]]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _perplexity(model, tokenizer, device, example: TrainingExample) -> float:
    messages = example.to_chat_messages()
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    return math.exp(outputs.loss.item())


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


class ModelEvaluator:
    """Evaluate a fine-tuned (merged) model against held-out data and a baseline."""

    def __init__(self, model_path: Path):
        """Initialize evaluator with fine-tuned (merged) model directory."""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            logger.warning(f"Model not found: {model_path}")

    def evaluate_on_test_set(self, test_data_path: Path) -> dict:
        """Evaluate model on test dataset: exact/normalized match, similarity, perplexity."""
        logger.info(f"Evaluating on test set: {test_data_path}")
        examples = ExampleLoader.load_dataset(Path(test_data_path))
        model, tokenizer, device = _load_model_and_tokenizer(str(self.model_path))

        exact_matches, normalized_matches, similarities, perplexities = [], [], [], []
        predictions = []

        for example in examples:
            prediction = _generate(model, tokenizer, device, example)
            predictions.append(prediction)

            exact_matches.append(prediction == example.output)
            normalized_matches.append(
                prediction.strip().lower() == example.output.strip().lower()
            )
            similarities.append(_similarity(prediction, example.output))
            perplexities.append(_perplexity(model, tokenizer, device, example))

        n = len(examples) or 1
        return {
            "status": "completed",
            "num_examples": len(examples),
            "exact_match_accuracy": sum(exact_matches) / n,
            "normalized_match_accuracy": sum(normalized_matches) / n,
            "avg_similarity": sum(similarities) / n,
            "avg_perplexity": sum(perplexities) / n,
            "predictions": predictions,
        }

    def compare_with_baseline(self, baseline_model: str, test_data_path: Path) -> dict:
        """Compare fine-tuned model against an un-tuned baseline model on the same test set."""
        logger.info(f"Comparing with baseline: {baseline_model}")

        finetuned_metrics = self.evaluate_on_test_set(test_data_path)

        baseline_evaluator = ModelEvaluator(Path(baseline_model))
        baseline_metrics = baseline_evaluator.evaluate_on_test_set(test_data_path)

        delta = {
            key: finetuned_metrics[key] - baseline_metrics[key]
            for key in (
                "exact_match_accuracy",
                "normalized_match_accuracy",
                "avg_similarity",
                "avg_perplexity",
            )
        }

        return {
            "status": "completed",
            "finetuned": finetuned_metrics,
            "baseline": baseline_metrics,
            "delta": delta,
        }

    def generate_report(self, output_path: Path, results: Optional[dict] = None) -> None:
        """Write evaluation results as JSON and a short Markdown summary."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating report: {output_path}")

        results = results or {}
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        md_path = output_path.with_suffix(".md")
        lines = ["# Evaluation Report", ""]
        if "finetuned" in results:
            lines.append("| Metric | Fine-tuned | Baseline | Delta |")
            lines.append("|---|---|---|---|")
            for key in results.get("delta", {}):
                lines.append(
                    f"| {key} | {results['finetuned'][key]:.4f} | "
                    f"{results['baseline'][key]:.4f} | {results['delta'][key]:+.4f} |"
                )
        else:
            for key, value in results.items():
                if key != "predictions":
                    lines.append(f"- **{key}**: {value}")

        md_path.write_text("\n".join(lines))
