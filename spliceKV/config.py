"""
推理加速所需要的配置文件所在地
"""


from dataclasses import dataclass  # 数据类装饰器，简化配置类定义

@dataclass(slots=True)
class Config:
    model: str 