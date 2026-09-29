"""
本模块是负责模型的前向传播
已完成：
加载模型权重；
对调度器选出的序列执行一次前向推理，为每个序列采样出下一个 token；

待完成：
分配和管理 PagedAttention 所需的 KV Cache（物理块分配）。当前每步全量重算，简单但低效；
支持并行处理（batch 前向、张量并行）
"""
import torch
from transformers import AutoModelForCausalLM

from spliceKV.config import Config

class ModelRunner:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model, torch_dtype="auto"
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def execute_model(self, scheduled):
        """对调度器选出的每个序列执行一次前向推理，返回各序列新产生的 token。"""
        next_tokens = []
        for seq in scheduled:
            input_ids = torch.tensor(
                [seq.prompt_token_ids + seq.output_token_ids], device=self.device
            )
            logits = self.model(input_ids).logits[:, -1, :]  # 最后一个位置的下一 token 预测
            next_tokens.append(self._sample(logits, seq.sampling_params))
        return next_tokens

    def _sample(self, logits, sampling_params):
        if sampling_params.temperature == 0:
            return logits.argmax(dim=-1).item()
        probs = torch.softmax(logits / sampling_params.temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
