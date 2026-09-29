# SpliceKV

## 目录结构

```
├── spliceKV/
│   ├── engine/
│   │   ├── llm_engine.py    # 引擎主体：请求入口 add_request、生成主循环 generate
│   │   ├── model_runner.py  # 模型前向：加载权重、执行推理、采样下一个 token
│   │   ├── schduler.py      # 调度器：waiting / running / finished 队列管理
│   │   └── sequence.py      # Sequence：封装一个生成请求（prompt、已生成 token、采样参数）
│   ├── models/              # （预留）模型相关实现
│   ├── config.py            # 推理配置
│   ├── llm.py               # 对外统一入口 LLM
│   └── sampling_params.py   # 采样参数
├── test/                    # （预留）测试
├── qwen3.5_download.ipynb   # 模型下载脚本
└── qwen3.5_test.py          # 端到端示例
```

## 环境

依赖：Python 3.14、PyTorch、transformers

先用 `qwen3.5_download.ipynb` 下载模型到本地，然后运行qwen3.5_test.py



