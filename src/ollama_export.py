"""Export a merged Hugging Face model to GGUF and register it with Ollama."""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"

# Matches TinyLlama-Chat's tokenizer_config.json chat_template / stop tokens.
TINYLLAMA_TEMPLATE = """TEMPLATE \"\"\"<|user|>
{{ .Prompt }}</s>
<|assistant|>
\"\"\"
PARAMETER stop "</s>"
PARAMETER stop "<|user|>"
"""


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    logger.info(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        logger.error(result.stdout)
        logger.error(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def ensure_llama_cpp(llama_cpp_dir: Path) -> Path:
    """Clone llama.cpp and install its conversion-script requirements if needed."""
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if convert_script.exists():
        logger.info(f"llama.cpp already present at {llama_cpp_dir}")
        return convert_script

    logger.info(f"Cloning llama.cpp into {llama_cpp_dir}...")
    llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(llama_cpp_dir)])

    requirements = llama_cpp_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"
    if not requirements.exists():
        requirements = llama_cpp_dir / "requirements.txt"
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])

    return convert_script


def convert_to_gguf(
    merged_model_dir: Path,
    output_gguf: Path,
    llama_cpp_dir: Path,
    outtype: str = "q8_0",
) -> Path:
    """Convert a merged Hugging Face model directory into a single GGUF file."""
    convert_script = ensure_llama_cpp(Path(llama_cpp_dir))
    output_gguf = Path(output_gguf)
    output_gguf.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            str(convert_script),
            str(merged_model_dir),
            "--outfile",
            str(output_gguf),
            "--outtype",
            outtype,
        ]
    )

    logger.info(f"GGUF model written to {output_gguf}")
    return output_gguf


def create_ollama_model(
    name: str,
    gguf_path: Path,
    system_prompt: str = "You are a helpful assistant.",
    modelfile_path: Optional[Path] = None,
) -> str:
    """Write a Modelfile for the GGUF model and register it with `ollama create`."""
    gguf_path = Path(gguf_path).resolve()
    modelfile_path = Path(modelfile_path) if modelfile_path else gguf_path.parent / "Modelfile"

    modelfile_content = f'FROM {gguf_path}\n\nSYSTEM """{system_prompt}"""\n\n{TINYLLAMA_TEMPLATE}'
    modelfile_path.write_text(modelfile_content)

    _run(["ollama", "create", name, "-f", str(modelfile_path)])
    logger.info(f"Registered Ollama model: {name}")
    return name
