"""

作用：
已完成：
初始化模型配置（Config）和 Tokenizer；
接收用户的生成请求（add_request），将 prompt 封装为 Sequence 对象；
调用 ModelRunner 执行前向推理，并收集生成的 token；
提供高层的 generate() 接口，通过调度器（Scheduler）管理请求的 Prefill 和 Decode 阶段，
批量生成并返回解码后的文本；

待完成：
根据 tensor_parallel_size 启动多个 ModelRunner 进程，实现张量并行（Tensor Parallelism）；

"""

from dataclasses import fields
from transformers import AutoTokenizer

from spliceKV.engine.model_runner import ModelRunner
from spliceKV.engine.schduler import Schduler
from spliceKV.engine.sequence import Sequence
from spliceKV.config import Config

class LLMEngine:
    def __init__(self, model, **kwargs):
        """
        初始化 LLMEngine 实例。

        【功能描述】
        构建完整的推理引擎环境，包括配置解析、Tokenizer 加载和调度器创建。

        【参数说明】
        :param model: str 类型，HuggingFace 模型路径。用于加载模型权重和词表。
        :param kwargs: dict 类型，额外的配置参数。合法的参数名由 Config 的 dataclass 字段决定，
                       
        【关键实现思路】
        
        """
        #对输入的kwargs进行键名匹配，筛选出属于 Config 的合法配置项
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}

        config = Config(model, **config_kwargs)

        #调用model_runner启动模型
        self.model_runner = ModelRunner(config)
        #加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        #加载调度器
        self.scheduler = Schduler(config)


    def add_request(self, prompt, sampling_params):
        """
        【功能描述】
        向推理引擎添加一个新的生成请求。将用户输入的 prompt 封装为 Sequence 对象，并提交给调度器管理。
        这是推理请求进入引擎的入口。

        【参数说明】
        :param prompt: str 或 list[int] 类型。
        :param sampling_params: SamplingParams 类型，控制生成行为的采样参数.

        """
        #如果prompt是字符串，则编码成token ID列表
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        #把请求封装为 Sequence 对象，交给调度器管理
        seq = Sequence(prompt, sampling_params, eos_token_id=self.tokenizer.eos_token_id)
        self.scheduler.add_request(seq)



    def generate(self, prompts, sampling_params, use_tqdm = False):
        #TODO tqdm加入
        #统一为列表，支持单个 prompt 输入
        if isinstance(prompts, str):
            prompts = [prompts]

        #将全部请求加入引擎
        for prompt in prompts:
            self.add_request(prompt, sampling_params)

        #调度循环：直到所有请求完成
        while self.scheduler.has_unfinished():
            scheduled = self.scheduler.schedule()              #选出本轮要执行的请求
            model_output = self.model_runner.execute_model(scheduled)  #前向推理，生成新 token
            self.scheduler.update(scheduled, model_output)     #更新请求状态，回收完成的请求

        #解码并返回结果
        outputs = [
            {"text": self.tokenizer.decode(seq.output_token_ids)}
            for seq in self.scheduler.get_finished()
        ]
        return outputs

