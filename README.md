# Fine-tuning a Small Llama Model with Ollama and LoRA

Workshop material for **PyCon Colombia 2026 (Medellín)**.

## Overview

This workshop teaches how to fine-tune a small Llama model using the **LoRA** (Low-Rank Adaptation) method and serve it locally with **Ollama**. The workshop runs on **Google Colab**, so attendees get free GPU access without any local setup.

By the end of the workshop, attendees will understand how to:

- Prepare and load a dataset for instruction fine-tuning.
- Fine-tune a small Llama model with LoRA using `transformers`, `peft`, and `accelerate` on a Colab GPU runtime.
- Export and run the fine-tuned model locally with Ollama.

## Requirements

- A Google account (to run the notebooks on [Google Colab](https://colab.research.google.com/))
- [Ollama](https://ollama.com/) installed locally to serve the fine-tuned model
- [uv](https://docs.astral.sh/uv/) if running the notebooks locally instead of on Colab

## Setup

### Google Colab (recommended)

Open the notebooks in `notebooks/` directly on Colab and run the setup cell at the top of each notebook — it installs dependencies via `pip`.

### Local (optional)

```bash
uv sync
uv run jupyter notebook
```

## Repository structure

- `notebooks/` — workshop notebooks, the main hands-on material, designed to run on Colab.
- `src/` — reusable Python modules used across notebooks and scripts.
- `scripts/` — standalone entry points (data prep, training, export to Ollama).
- `config/` — configuration files (LoRA hyperparameters, training settings).
