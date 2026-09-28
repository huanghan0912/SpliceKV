
import os
## 使用国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3.5-2B"  
CACHE_DIR = os.path.join(os.getcwd(), ".cache", "huggingface")
os.makedirs(CACHE_DIR, exist_ok=True)

# 1. 显式下载：带进度条、断点续传，保存到 ./.cache/huggingface
print(f"正在下载 {model_name} ...")
local_path = snapshot_download(
    repo_id=model_name,
    cache_dir=CACHE_DIR,
    max_workers=4,             # 弱网下 8 并发容易触发超时，降到 4 更稳
)
print(f"模型已下载到: {local_path}")

# 2. 直接从本地路径加载（纯离线，不走网络）
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(
    local_path,
    device_map="auto",         # 有 GPU 自动放 GPU，否则 CPU
    torch_dtype="auto",        # 自动选 bf16/fp16
)

# 2. 简单对话测试：用 chat template 构造输入
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "用2-3句话解释什么是 KV Cache。"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

# 3. 生成
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.8,
    do_sample=True,
)

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))