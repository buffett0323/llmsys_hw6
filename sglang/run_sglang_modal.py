"""
Run `run_sglang.py`-style AlpacaEval inference on Modal with 2× L40S.

Tensor parallel size must match GPU count: `tp_size=2` with `gpu="L40S:2"`.

Usage:
    modal run run_sglang_modal.py
    modal run run_sglang_modal.py --model-id Qwen/Qwen2.5-7B-Instruct-1M --output-file outputs.jsonl
"""

import json
import os

import modal

# ---------------------------------------------------------------------------
# Persistent volume — model weights cached here between runs.
# ---------------------------------------------------------------------------
model_vol = modal.Volume.from_name("sglang-model-weights", create_if_missing=True)
MODEL_DIR = "/models"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"

# Two L40S GPUs → use tensor parallel 2 (local run_sglang.py uses tp_size=8 on 8 GPUs).
_INFERENCE_GPU = "L40S:2"
_DEFAULT_TP_SIZE = 2

sglang_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("libnuma-dev")
    .pip_install(
        "sglang[all]>=0.3.0",
        "datasets",
        "transformers",
        "accelerate",
        "tqdm",
        "huggingface_hub",
    )
)

app = modal.App("sglang-alpaca-eval", image=sglang_image)


@app.function(
    volumes={MODEL_DIR: model_vol},
    timeout=3600,
)
def download_model(model_id: str = DEFAULT_MODEL):
    from huggingface_hub import snapshot_download

    local_path = os.path.join(MODEL_DIR, model_id.replace("/", "--"))
    if os.path.isdir(local_path) and os.listdir(local_path):
        print(f"Model already cached at {local_path}, skipping download.")
        return local_path
    print(f"Downloading {model_id} → {local_path} ...")
    snapshot_download(repo_id=model_id, local_dir=local_path)
    model_vol.commit()
    print("Download complete.")
    return local_path


@app.function(
    gpu=_INFERENCE_GPU,
    timeout=7200,
    volumes={MODEL_DIR: model_vol},
)
def run_inference(
    model_id: str = DEFAULT_MODEL,
    tp_size: int = _DEFAULT_TP_SIZE,
    mem_fraction_static: float = 0.8,
    batch_size: int = 16,
) -> list[dict]:
    """Same logic as `run_sglang.py`; model loaded from the volume."""
    from datasets import load_dataset
    import sglang as sgl
    from tqdm import tqdm

    local_path = os.path.join(MODEL_DIR, model_id.replace("/", "--"))
    if not os.path.isdir(local_path) or not os.listdir(local_path):
        raise RuntimeError(
            f"Model weights not found at {local_path}. "
            "Run `modal run run_sglang_modal.py::download_model` first."
        )

    dataset = load_dataset(
        "json",
        data_files={
            "eval": "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json",
        },
        split="eval",
    )

    llm = sgl.Engine(
        model_path=local_path,
        mem_fraction_static=mem_fraction_static,
        tp_size=tp_size,
        cuda_graph_max_bs=64,
        trust_remote_code=True,
        # Qwen2.5-*-1M uses dual chunk attention; flashinfer doesn't support it.
        attention_backend="dual_chunk_flash_attn",
        # Piecewise CUDA graph warmup produces tensor size mismatch (71680 != 14336)
        # with dual_chunk_flash_attn. Disable it as recommended by SGLang.
        disable_cuda_graph=True,
    )

    prompts = [row["instruction"] for row in dataset]
    sampling_params = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8192}

    outputs = []
    for start in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[start : start + batch_size]
        batch_outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        for output in batch_outputs:
            outputs.append(output)

    llm.shutdown()

    # Full jsonl (one line per prompt). Local script uses range(0, len, 10) which skips rows.
    return [
        {"output": outputs[i], "instruction": prompts[i]}
        for i in range(len(outputs))
    ]


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL,
    output_file: str = "outputs.jsonl",
    tp_size: int = _DEFAULT_TP_SIZE,
    mem_fraction_static: float = 0.8,
    batch_size: int = 16,
):
    download_model.remote(model_id=model_id)

    results = run_inference.remote(
        model_id=model_id,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,
        batch_size=batch_size,
    )

    with open(output_file, "w") as f:
        for record in results:
            f.write(json.dumps(record, default=str) + "\n")

    print(f"Wrote {len(results)} results to {output_file}")
