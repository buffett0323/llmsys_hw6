"""
Run the SGLang alpaca-eval inference job on Modal (GPU cloud).

Usage:
    # First run: downloads model weights into the persistent volume (~15 GB).
    # Subsequent runs skip the download and start immediately.
    modal run run_sglang_modal.py

    # Override model / output file:
    modal run run_sglang_modal.py --model-path Qwen/Qwen2.5-7B-Instruct \
                                   --output-file my_outputs.jsonl
"""

import modal
import json
import os

# ---------------------------------------------------------------------------
# Persistent volume — model weights are downloaded here once and reused.
# ---------------------------------------------------------------------------
model_vol = modal.Volume.from_name("sglang-model-weights", create_if_missing=True)
MODEL_DIR = "/models"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"

# ---------------------------------------------------------------------------
# Image — CUDA 12.8 devel (matches Modal A100 runtime) + libnuma for sgl_kernel.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Download function — run once to populate the volume, then cached forever.
# Call explicitly with:  modal run run_sglang_modal.py::download_model
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------
# To use a gated model, first create a Modal secret:
#   modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_...
# then add  secrets=[modal.Secret.from_name("huggingface-secret")]  below.
@app.function(
    gpu="A100-80GB",
    timeout=7200,
    volumes={MODEL_DIR: model_vol},
)
def run_inference(
    model_id: str = DEFAULT_MODEL,
    dp_size: int = 1,
    tp_size: int = 1,
    mem_fraction_static: float = 0.80,
    batch_size: int = 256,
) -> list[dict]:
    """Run SGLang offline inference on the AlpacaEval benchmark."""
    from datasets import load_dataset
    import sglang as sgl
    from transformers import AutoTokenizer
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
        dp_size=dp_size,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,
        # Required for long-context models that use dual chunk attention
        # (e.g. Qwen2.5-*-1M).  flashinfer (the default) does not support it.
        attention_backend="dual_chunk_flash_attn",
        # Cap CUDA graph capture batch sizes to avoid OOM during init.
        cuda_graph_max_bs=4,
    )

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    raw_prompts = [item["instruction"] for item in dataset]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in raw_prompts
    ]

    sampling_params = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8192}

    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i : i + batch_size]
        batch_outputs = llm.generate(batch, sampling_params)
        for out in batch_outputs:
            outputs.append(out["text"])

    llm.shutdown()

    return [
        {"output": outputs[i], "instruction": raw_prompts[i]}
        for i in range(len(outputs))
    ]


# ---------------------------------------------------------------------------
# Local entrypoint — downloads model if needed, then runs inference.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL,
    output_file: str = "outputs.jsonl",
    dp_size: int = 1,
    tp_size: int = 1,
    mem_fraction_static: float = 0.80,
    batch_size: int = 256,
):
    # Ensure weights are present in the volume before inference starts.
    download_model.remote(model_id=model_id)

    results = run_inference.remote(
        model_id=model_id,
        dp_size=dp_size,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,
        batch_size=batch_size,
    )

    with open(output_file, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(results)} results to {output_file}")
