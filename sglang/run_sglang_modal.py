"""
Modal version of run_sglang.py — runs SGLang inference on a Modal A100 GPU.

Usage:
    modal run run_sglang_modal.py
    modal run run_sglang_modal.py --model-path Qwen/Qwen2.5-7B-Instruct-1M
    modal run run_sglang_modal.py --model-path Qwen/Qwen2.5-7B-Instruct-1M --batch-size 32
"""

import json
import os

import modal

# ---------------------------------------------------------------------------
# Modal image — install SGLang and dependencies
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sglang[all]>=0.4.0",
        "datasets",
        "huggingface_hub",
        "tqdm",
        "torch",
        extra_options="--find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/",
    )
)

app = modal.App("sglang-inference", image=image)

# Persistent volume to cache downloaded model weights across runs
model_cache = modal.Volume.from_name("sglang-model-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    timeout=60 * 60,           # 1 hour — large models can take time
    volumes={"/model-cache": model_cache},
    # secrets=[modal.Secret.from_name("huggingface-secret")],  # uncomment if using gated models
)
def run_inference(
    model_path: str = "Qwen/Qwen2.5-7B-Instruct-1M",
    batch_size: int = 16,
    mem_fraction_static: float = 0.85,
) -> list[dict]:
    """Run SGLang inference on alpaca_eval and return results as a list of dicts."""
    import sglang as sgl
    from datasets import load_dataset
    from tqdm import tqdm

    # Point HF cache at the persistent volume so weights are reused
    os.environ["HF_HOME"] = "/model-cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache/hf"

    print(f"Loading alpaca_eval dataset ...")
    dataset = load_dataset(
        "json",
        data_files={
            "eval": "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json",
        },
        split="eval",
    )

    prompts = [row["instruction"] for row in dataset]
    print(f"Loaded {len(prompts)} prompts.")

    print(f"Initialising SGLang engine with model: {model_path}")
    llm = sgl.Engine(
        model_path=model_path,
        mem_fraction_static=mem_fraction_static,
        trust_remote_code=True,
        attention_backend="dual_chunk_flash_attn",
    )

    sampling_params = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8192}
    outputs = []
    batch_size = max(1, batch_size)

    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[start : start + batch_size]
            result = llm.generate(prompt=batch, sampling_params=sampling_params)
            texts = _texts_from_generate(result)
            if len(texts) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} outputs, got {len(texts)} from generate()"
                )
            outputs.extend(texts)
    finally:
        llm.shutdown()

    return [
        {"instruction": instruction, "output": output}
        for instruction, output in zip(prompts, outputs)
    ]


def _texts_from_generate(result):
    """Normalize Engine.generate return value across single/batch prompts."""
    if isinstance(result, dict):
        return [result.get("text", "")]
    if isinstance(result, list):
        out = []
        for item in result:
            if isinstance(item, dict):
                out.append(item.get("text", ""))
            else:
                out.append(str(item))
        return out
    return [str(result)]


# ---------------------------------------------------------------------------
# Local entrypoint — call the remote function and save results
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    model_path: str = "Qwen/Qwen2.5-7B-Instruct-1M",
    output_file: str = "outputs.jsonl",
    batch_size: int = 16,
    mem_fraction_static: float = 0.85,
):
    print(f"Submitting job to Modal (GPU: A100) ...")
    results = run_inference.remote(
        model_path=model_path,
        batch_size=batch_size,
        mem_fraction_static=mem_fraction_static,
    )

    with open(output_file, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(results)} outputs to {output_file}")
