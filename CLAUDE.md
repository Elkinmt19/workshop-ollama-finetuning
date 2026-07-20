# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Workshop material for PyCon Colombia 2026 (Medellín): teaches how to fine-tune a small Llama model using **Ollama** and the **LoRA** method. The workshop runs on **Google Colab** (free GPU runtime, no local setup for attendees); local execution is a secondary path for authoring/testing content.

## Current state

This repository is a scaffold. `config/`, `notebooks/`, `scripts/`, and `src/` exist but are empty — there is no code, no tests, and no lint/formatter config yet. Only `pyproject.toml` is populated. Treat any architectural description here as provisional until real content lands; update this file as the workshop content is built out.

## Environment & commands

Dependency management is **uv** (`[tool.uv] package = false` — this is not a distributable package, just a workshop environment). Python `>=3.12`, standard PEP 621 `[project]` metadata (no Poetry).

```bash
uv sync                # install all deps (see pyproject.toml)
uv run jupyter notebook   # or: uv run jupyter lab
```

On **Google Colab** (the primary target platform), notebooks install their own deps via a `pip install` cell at the top rather than relying on `uv`/`pyproject.toml` — Colab runtimes are ephemeral and don't share this environment.

No test suite, linter, or formatter is configured yet — don't assume `pytest`/`ruff`/`black` exist until they're added to `pyproject.toml`.

## Stack

- **Ollama**: `ollama` Python client — for running/serving the fine-tuned small Llama model locally.
- **Fine-tuning**: `torch`, `transformers`, `peft`, `accelerate` — LoRA fine-tuning stack, run on Colab's GPU runtime.
- **Config/validation**: `pydantic`, `pydantic-settings`, `python-dotenv`.
- **Notebook/data**: `jupyter`, `notebook`, `ipywidgets`, `pandas`, `numpy`, `matplotlib`.

## Intended structure (folders exist, currently empty)

- `notebooks/` — workshop notebooks (likely the primary teaching artifact given the Jupyter-heavy dependency set).
- `src/` — reusable Python modules imported by notebooks/scripts.
- `scripts/` — standalone/CLI entry points (e.g. data prep, training, export to Ollama).
- `config/` — configuration files (e.g. LoRA hyperparameters, training settings).
