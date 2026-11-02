"""GPU smoke test: let the untrained Qwen2.5-3B answer a few training questions with vLLM.

Checks that vLLM + the GPU work, and shows how the base model behaves with the Search-R1 prompt
(does it emit <tool_call> search queries / <answer> tags before any RL?).

Usage (verl env, from the verl repo dir):
  uv run --frozen --all-packages --extra vllm --extra fsdp python \
      /root/autodl-tmp/search-r1-judge-hacking/scripts/smoke_generate.py
"""
import argparse

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

p = argparse.ArgumentParser()
p.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct")
p.add_argument("--data", default="/root/autodl-tmp/data/searchR1_processed_direct/train.parquet")
p.add_argument("--n", type=int, default=5)
p.add_argument("--gpu_mem", type=float, default=0.4)
args = p.parse_args()

df = pd.read_parquet(args.data).sample(args.n, random_state=0)
tok = AutoTokenizer.from_pretrained(args.model)
prompts = [tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True) for m in df["prompt"]]

llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096, dtype="bfloat16")
sp = SamplingParams(temperature=0.0, max_tokens=400, stop=["</tool_call>", "</answer>"], include_stop_str_in_output=True)

for (_, row), out in zip(df.iterrows(), llm.generate(prompts, sp)):
    text = out.outputs[0].text
    print("=" * 100)
    print("Q:", row["extra_info"]["question"])
    print("gold:", row["reward_model"]["ground_truth"]["target"] if row["reward_model"] is not None else "?")
    print("-" * 100)
    print(text)
    tag = "SEARCH" if "<tool_call>" in text else ("ANSWER" if "<answer>" in text else "NEITHER")
    print(f"--> first action: {tag}")
