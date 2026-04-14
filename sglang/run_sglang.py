from datasets import load_dataset
import sglang as sgl
import asyncio
import json
import argparse

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
    args = parser.parse_args()

    dataset = load_dataset(
        "json",
        data_files={
            "eval": "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json",
        },
        split="eval",
    )
    model_path = args.model_path

    # TODO: initialize sglang egnine here
    # you may want to explore different args we can pass here to make the inference faster
    # e.g. dp_size, mem_fraction_static
    llm = sgl.Engine(
        model_path=model_path,
        mem_fraction_static=0.8,  
        tp_size=8,    
        cuda_graph_max_bs=64,
        trust_remote_code=True,
    )

    prompts = []

    for i in dataset:
        prompts.append(i['instruction'])

    sampling_params = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 8192}

    outputs = []

    # TODO: you may want to explore different batch_size
    batch_size = 16 #len(prompts) 

    from tqdm import tqdm
    for i in tqdm(range(0, len(prompts), batch_size)):
        # TODO: prepare the batched prompts and use llm.generate
        # save the output in outputs
        batch_prompts = prompts[i:i+batch_size]
        
        # Generate responses for the batch
        batch_outputs = llm.generate(
            batch_prompts,
            sampling_params=sampling_params
        )
        
        # Extract the generated text from the outputs
        for output in batch_outputs:
            outputs.append(output)

    with open(args.output_file, "w") as f:
        for i in range(0, len(outputs), 10):
            instruction = prompts[i]
            output = outputs[i]
            f.write(json.dumps({
                "output": output,
                "instruction": instruction
            }) + "\n")

if __name__ == "__main__":
    main()