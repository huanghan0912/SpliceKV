"""
调度器
"""

from spliceKV.config import Config


class Schduler:
    def __init__(self, config: Config):
        self.config = config
        self.waiting = []    # 等待调度的请求
        self.running = []    # 正在运行的请求
        self.finished = []   # 已完成的请求

    def add_request(self, seq):
        """将请求放入等待队列。"""
        self.waiting.append(seq)

    def has_unfinished(self):
        """是否还有未完成的请求。"""
        return bool(self.waiting or self.running)

    def schedule(self):
        """从等待队列中挑选请求，返回本次需要执行前向计算的序列。

        简单实现：每轮把等待队列中的请求全部转入运行队列（执行 Prefill），
        与已在运行的请求（继续 Decode）一起返回。
        """
        for seq in self.waiting:
            self.running.append(seq)
        self.waiting = []
        return self.running

    def update(self, scheduled, model_output):
        """根据模型输出更新序列状态，回收已完成的请求。"""
        still_running = []
        for seq, next_token in zip(scheduled, model_output):
            seq.output_token_ids.append(next_token)
            if seq.is_finished():
                self.finished.append(seq)
            else:
                still_running.append(seq)
        self.running = still_running

    def get_finished(self):
        """按请求提交顺序返回已完成的序列。"""
        return self.finished
