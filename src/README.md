# `src/` — 源码

```
src/
├── config.py                # 路径/模型 ID 解析（按语言）
├── data/                    # 语料加载、prefix 切分、配对样本
├── extraction/              # 隐藏状态抽取
├── generation/              # 文本生成（API + 本地，可加引导 hook）
├── steering/                # v_src 学习、SAE、可解释性
├── evaluation/              # 引导模型推理包装
└── visualization/           # 全部 11+8 张绘图函数
```

各子目录的 README 详细说明对应模块的接口和公式。
