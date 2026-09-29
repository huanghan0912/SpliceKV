import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from spliceKV.llm import LLM
from spliceKV.sampling_params import SamplingParams

model_name = "Qwen/Qwen3.5-2B"
model_path = ".cache/qwen3.5-2B"

#加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)


llm = LLM(model_path)


#构建几个问题
prompts = [
        "介绍一下你自己",
        "列举100以内的所有素数",
        "用2-3句话解释什么是 KV Cache。"
    ]

prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)


for prompt, output in zip(prompts, outputs):
    print("\n")
    print(f"Prompt: {prompt!r}")
    print(f"Completion: {output['text']!r}")


