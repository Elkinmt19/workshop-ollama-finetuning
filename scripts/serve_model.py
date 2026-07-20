#!/usr/bin/env python3
"""Export a merged fine-tuned model to GGUF and serve it via Ollama."""

import argparse
import logging
from pathlib import Path

import requests

from src.ollama_export import convert_to_gguf, create_ollama_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for GGUF export + Ollama registration."""
    parser = argparse.ArgumentParser(description="Serve a fine-tuned model with Ollama")
    parser.add_argument(
        "--merged-model-dir",
        type=Path,
        required=True,
        help="Directory containing the merged Hugging Face model",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="tinyllama-finetuned:latest",
        help="Ollama model tag to register",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default="llama.cpp",
        help="Directory to clone/use llama.cpp for GGUF conversion",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        default="q8_0",
        help="GGUF quantization/output type",
    )
    parser.add_argument(
        "--test-prompt",
        type=str,
        help="If provided, sends this prompt to the served model after registration",
    )

    args = parser.parse_args()

    gguf_path = args.merged_model_dir / "model.gguf"
    convert_to_gguf(args.merged_model_dir, gguf_path, args.llama_cpp_dir, outtype=args.outtype)
    create_ollama_model(args.name, gguf_path)

    logger.info(f"Model '{args.name}' registered with Ollama.")

    if args.test_prompt:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": args.name, "prompt": args.test_prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        logger.info(f"Test generation: {response.json()['response']}")


if __name__ == "__main__":
    main()
