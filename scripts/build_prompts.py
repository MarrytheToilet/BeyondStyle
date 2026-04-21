#!/usr/bin/env python3
"""
Build neutral prompt prefixes from a held-out corpus, for use in steering
inference and evaluation. Output format:

    [
      "<prefix-1>",
      "<prefix-2>",
      ...
    ]

The corpus should be DIFFERENT from the one used for the paired training
dataset, so that the steering vector v_src is evaluated on truly unseen
text. Defaults:

    EN paired:   CNN/DailyMail   →   prompts default: Wikipedia
    ZH paired:   THUCNews        →   prompts default: Wikipedia (zh)

Usage:
    # English from Wikipedia (different from CNN/DailyMail)
    python scripts/build_prompts.py --language en --corpus wikipedia \
        --num-prompts 100

    # Chinese from Wikipedia
    python scripts/build_prompts.py --language zh --corpus wikipedia \
        --num-prompts 100

    # Custom output path
    python scripts/build_prompts.py -l en -c ag_news -n 50 \
        -o data/prompts/en/agnews_50.json
"""

import sys
import json
import argparse
import logging
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corpus import CorpusLoader
from src.data.splitter import PrefixSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _is_clean_prefix(text: str, language: str, min_chars: int) -> bool:
    if len(text) < min_chars:
        return False
    if language == "zh":
        cjk_chars = len(CJK_RE.findall(text))
        if cjk_chars < 0.5 * len(text):
            return False
    else:
        if len(CJK_RE.findall(text)) > 5:
            return False
    alpha = sum(1 for c in text if c.isalpha() or CJK_RE.match(c))
    if alpha < 0.5 * len(text):
        return False
    return True


def _trim_to_sentence_boundary(text: str, language: str) -> str:
    """Trim trailing partial sentence so prefix ends at a natural pause."""
    if language == "zh":
        for stop in ["，", "。", "；", "！", "？"]:
            idx = text.rfind(stop, max(0, len(text) - 50))
            if idx > 0:
                return text[: idx + 1]
    else:
        for stop in [". ", ", ", "; ", "! ", "? "]:
            idx = text.rfind(stop, max(0, len(text) - 80))
            if idx > 0:
                return text[: idx + len(stop)].rstrip()
    return text


def _load_corpus(loader: CorpusLoader, corpus: str, language: str, max_docs: int):
    if corpus == "wikipedia":
        return loader.load_wikipedia(language=language, max_docs=max_docs)
    if corpus == "ag_news":
        return loader.load_ag_news(max_docs=max_docs)
    if corpus == "cnn_dailymail":
        return loader.load_cnn_dailymail(max_docs=max_docs)
    if corpus == "chinese_news":
        return loader.load_chinese_news(max_docs=max_docs)
    raise ValueError(f"Unknown corpus: {corpus}")


def build_prompts(
    language: str,
    corpus: str,
    num_prompts: int,
    prefix_tokens: int,
    max_docs: int,
) -> list:
    loader = CorpusLoader(language=language)
    docs = _load_corpus(loader, corpus, language, max_docs)

    splitter = PrefixSplitter(
        target_prefix_tokens=prefix_tokens,
        target_continuation_tokens=50,  # discarded
        tolerance=0.3,
    )

    min_chars = 80 if language == "en" else 40
    prompts = []
    seen = set()

    try:
        from tqdm import tqdm
        bar = tqdm(total=num_prompts, desc=f"Collecting {language} prompts")
    except ImportError:
        bar = None

    for doc in docs:
        if len(prompts) >= num_prompts:
            break
        try:
            pairs = splitter.split_text(
                text=doc.text, doc_id=doc.doc_id,
                domain=doc.domain, language=doc.language,
                max_pairs=1,
            )
        except Exception:
            continue
        for pair in pairs:
            if len(prompts) >= num_prompts:
                break
            prefix = pair.prefix.strip()
            if not _is_clean_prefix(prefix, language, min_chars):
                continue
            prefix = _trim_to_sentence_boundary(prefix, language)
            if not _is_clean_prefix(prefix, language, min_chars):
                continue
            key = prefix[:60]
            if key in seen:
                continue
            seen.add(key)
            prompts.append(prefix)
            if bar is not None:
                bar.update(1)

    if bar is not None:
        bar.close()

    logger.info(f"Built {len(prompts)} prompts from {corpus} ({language})")
    return prompts


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--language", "-l", choices=["en", "zh"], required=True)
    p.add_argument("--corpus", "-c", default="wikipedia",
                   choices=["wikipedia", "ag_news", "cnn_dailymail", "chinese_news"])
    p.add_argument("--num-prompts", "-n", type=int, default=100)
    p.add_argument("--prefix-tokens", type=int, default=80)
    p.add_argument("--max-docs", type=int, default=3000)
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Output JSON path. Default: data/prompts/<lang>/<corpus>_<n>.json")
    return p.parse_args()


def main():
    args = parse_args()
    prompts = build_prompts(
        language=args.language,
        corpus=args.corpus,
        num_prompts=args.num_prompts,
        prefix_tokens=args.prefix_tokens,
        max_docs=args.max_docs,
    )
    if args.output is None:
        out_dir = Path("data/prompts") / args.language
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.corpus}_{len(prompts)}.json"
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(prompts)} prompts to {out_path}")


if __name__ == "__main__":
    main()
