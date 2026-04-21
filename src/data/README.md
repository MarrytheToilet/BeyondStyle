# `src/data/` — 数据加载与配对样本

## 模块

| 文件 | 作用 |
|------|------|
| `corpus.py` | `CorpusLoader`：从 HuggingFace 下载/缓存 CNN/DailyMail、Wikipedia、AG News、THUCNews（中文） |
| `splitter.py` | `PrefixSplitter`：按 token 数切 prefix / continuation |
| `paired_dataset.py` | `PairedSample` dataclass + JSON 序列化 |
| `continuation_gen.py` | `ContinuationGenerator`：调用 backend 生成 AI 续写 |
| `prompt_builder.py` | `PromptBuilder`：根据 prefix + 主题/实体生成轻约束 prompt |

## 用法

```python
from src.data.corpus import CorpusLoader
from src.data.splitter import PrefixSplitter

loader = CorpusLoader(language="zh")
docs = loader.load("chinese_news", max_docs=2000)
splitter = PrefixSplitter(prefix_tokens=100, continuation_tokens=200, language="zh")
pairs = splitter.split(docs)
```

## 中文支持

- `corpus.py` 通过 `language="zh"` 触发 THUCNews 路径
- `splitter.py` 在中文模式下用 `tiktoken` 估计字符到 token 的换算
- `prompt_builder.py` 中文模板使用「请你按照原文风格继续写下去」等中性指令
