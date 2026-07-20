#!/usr/bin/env python3
"""Fine-tune TinyLlama with LoRA, ready for later Ollama export."""

import argparse
import logging
from pathlib import Path

from src.trainer import OllamaTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model with LoRA")
    parser.add_argument(
        "--config",
        type=Path,
        default="config/training_config.yaml",
        help="Training configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Base model HF id (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)",
    )

    args = parser.parse_args()

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return

    logger.info("Initializing trainer...")
    trainer = OllamaTrainer(args.config)

    if args.model:
        trainer.model_name = args.model
        trainer.config["model"]["base_model_hf"] = args.model
    if args.epochs:
        trainer.config["training"]["epochs"] = args.epochs
    if args.batch_size:
        trainer.config["training"]["batch_size"] = args.batch_size

    logger.info(trainer.get_config_summary())

    result = trainer.train()

    logger.info("Fine-tuning complete:")
    logger.info(f"  Final loss: {result['final_loss']}")
    logger.info(f"  Merged model: {result['merged_dir']}")
    if result.get("adapter_dir"):
        logger.info(f"  LoRA adapter: {result['adapter_dir']}")


if __name__ == "__main__":
    main()
