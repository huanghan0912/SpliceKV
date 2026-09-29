"""
推理采样参数
"""

from dataclasses import dataclass 


@dataclass(slots=True)
class SamplingParams:
    

    temperature: float = 1.0  
    max_tokens: int = 300      

    