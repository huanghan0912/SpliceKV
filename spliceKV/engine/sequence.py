"""
最小可用的请求序列对象
"""

class Sequence:
    """封装一个生成请求：prompt token、已生成的 token 和采样参数。"""

    def __init__(self, prompt_token_ids, sampling_params, eos_token_id=None):
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids = []
        self.sampling_params = sampling_params
        self.eos_token_id = eos_token_id

    def is_finished(self):
        #生成到 EOS 或达到最大生成长度即结束
        if self.eos_token_id is not None and self.output_token_ids \
                and self.output_token_ids[-1] == self.eos_token_id:
            return True
        return len(self.output_token_ids) >= self.sampling_params.max_tokens
