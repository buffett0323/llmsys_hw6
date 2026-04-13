import json
import os
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent


def _fix_sys_path_for_sglang_package() -> None:
    """This homework lives in a directory named ``sglang/``. If the repo root
    is on ``sys.path`` (common when running from the project root), Python can
    treat ``<repo>/sglang`` as the top-level ``sglang`` package and shadow the
    real ``sglang`` library from pip."""
    while str(_repo_root) in sys.path:
        sys.path.remove(str(_repo_root))
    if sys.path[:1] == [""] and Path.cwd().resolve() == _repo_root:
        sys.path.pop(0)
    os.chdir(_script_dir)


_fix_sys_path_for_sglang_package()

import argparse

from datasets import load_dataset
from tqdm import tqdm

import sglang as sgl


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


def main():
    parser = argparse.ArgumentParser(description="Run inference with a specific model path.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs.jsonl",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of prompts per generate() call.",
    )
    parser.add_argument(
        "--mem_fraction_static",
        type=float,
        default=0.85,
        help="Fraction of GPU memory reserved for static allocations (KV etc.).",
    )
    args = parser.parse_args()

    dataset = load_dataset(
        "json",
        data_files={
            "eval": "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json",
        },
        split="eval",
    )
    model_path = args.model_path

    llm = sgl.Engine(
        model_path=model_path,
        mem_fraction_static=args.mem_fraction_static,
        trust_remote_code=True,
    )

    prompts = [row["instruction"] for row in dataset]

    sampling_params = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8192}

    outputs = []
    batch_size = max(1, args.batch_size)

    try:
        for start in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[start:start + batch_size]
            result = llm.generate(prompt=batch, sampling_params=sampling_params)
            texts = _texts_from_generate(result)
            if len(texts) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} outputs, got {len(texts)} from generate()"
                )
            outputs.extend(texts)
    finally:
        llm.shutdown()

    with open(args.output_file, "w") as f:
        for instruction, output in zip(prompts, outputs):
            f.write(
                json.dumps(
                    {
                        "output": output,
                        "instruction": instruction,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
