from datasets import load_dataset
import sglang as sgl
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer


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
        "--dp_size",
        type=int,
        default=1,
        help="Data parallelism size (number of GPUs for DP)",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size (number of GPUs for TP)",
    )
    parser.add_argument(
        "--mem_fraction_static",
        type=float,
        default=0.85,
        help="Fraction of GPU memory reserved for static KV cache",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of prompts per generate call",
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

    # Initialize SGLang offline engine.
    # dp_size / tp_size let you spread across multiple GPUs;
    # mem_fraction_static controls how much VRAM is pre-allocated for the KV cache.
    llm = sgl.Engine(
        model_path=model_path,
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        # Required for long-context models that use dual chunk attention
        # (e.g. Qwen2.5-*-1M).  flashinfer (the default) does not support it.
        attention_backend="dual_chunk_flash_attn",
    )

    # Apply the model's chat template so instruction-following works correctly.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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

    batch_size = args.batch_size
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i : i + batch_size]
        batch_outputs = llm.generate(batch, sampling_params)
        # Each element is a dict with key "text" containing the generated string.
        for out in batch_outputs:
            outputs.append(out["text"])

    with open(args.output_file, "w") as f:
        for i in range(len(outputs)):
            f.write(json.dumps({
                "output": outputs[i],
                "instruction": raw_prompts[i],
            }) + "\n")

    llm.shutdown()


if __name__ == "__main__":
    main()

