#!/usr/bin/env python3
"""
Build a large-scale paired dataset for learning v_src direction.

This script:
1. Loads real corpus data from CNN/DailyMail (default) or other sources via HuggingFace
2. Splits documents into prefix-continuation pairs (token-based)
3. Generates AI continuations via configurable backend (API or local)
4. Saves paired dataset with dual continuations (guided + free)

Usage:
    # Default: CNN/DailyMail corpus with .env configuration
    python scripts/build_dataset.py -n 300

    # Override backend from CLI
    python scripts/build_dataset.py --backend siliconflow --num-samples 500

    # Use local corpus JSONL instead of downloading
    python scripts/build_dataset.py --corpus-jsonl data/raw/my_corpus.jsonl

    # AG News or Wikipedia (alternative corpora)
    python scripts/build_dataset.py --corpus ag_news -n 300
    python scripts/build_dataset.py --corpus wikipedia -n 50

    # Quick test with small sample
    python scripts/build_dataset.py --num-samples 20 --max-docs 50
"""

import sys
import os
import argparse
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

# Allow running from scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.config import get_model_id, get_dataset_dir
from src.data.corpus import CorpusLoader
from src.data.splitter import PrefixSplitter
from src.data.paired_dataset import PairedSample
from src.data.continuation_gen import ContinuationGenerator
from src.data.prompt_builder import PromptBuilder
from src.generation.backends import create_generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Defaults (overridable via CLI or .env)
# ============================================================
DEFAULT_NUM_SAMPLES = 500
DEFAULT_MAX_WIKI_DOCS = 2000
DEFAULT_MAX_WIKIHOW_DOCS = 500
DEFAULT_LANGUAGE = "en"

# Token targets — Wikipedia / WikiHow (longer documents)
WIKI_PREFIX_TOKENS = 128
WIKI_CONTINUATION_TOKENS = 256

# Token targets — AG News (shorter, alternative corpus)
AGNEWS_PREFIX_TOKENS = 50
AGNEWS_CONTINUATION_TOKENS = 80

# Token targets — CNN/DailyMail (full-length news articles)
CNN_PREFIX_TOKENS = 100
CNN_CONTINUATION_TOKENS = 200

# Default filtering for AG News
AGNEWS_MIN_WORDS = 60
AGNEWS_MAX_WORDS = 300

# Default filtering for CNN/DailyMail
CNN_MIN_WORDS = 200
CNN_MAX_WORDS = 600

TOKEN_TOLERANCE = 0.20


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Build paired human/AI continuation dataset from real corpora."
    )
    # Backend
    p.add_argument("--backend", "-b", type=str, default=None,
                   help="Generation backend: siliconflow, openai, local (default: from .env)")
    p.add_argument("--model-name", type=str, default=None,
                   help="Model name for API backends (default: from .env)")
    p.add_argument("--model-path", "-m", type=str, default=None,
                   help="Local model path (only for --backend local)")
    p.add_argument("--api-key", type=str, default=None,
                   help="API key (default: from .env)")

    # Data source
    p.add_argument("--corpus", type=str, default="cnn_dailymail",
                   choices=["wikipedia", "ag_news", "cnn_dailymail"],
                   help="Corpus source (default: cnn_dailymail)")
    p.add_argument("--corpus-jsonl", type=str, default=None,
                   help="Load corpus from local JSONL instead of downloading")
    p.add_argument("--max-docs", type=int, default=DEFAULT_MAX_WIKI_DOCS,
                   help=f"Max Wikipedia documents to load (default: {DEFAULT_MAX_WIKI_DOCS})")
    p.add_argument("--max-wikihow-docs", type=int, default=DEFAULT_MAX_WIKIHOW_DOCS,
                   help=f"Max WikiHow documents to load (default: {DEFAULT_MAX_WIKIHOW_DOCS})")
    p.add_argument("--language", "-l", type=str, default=DEFAULT_LANGUAGE,
                   help=f"Language code (default: {DEFAULT_LANGUAGE})")

    # AG News specific
    p.add_argument("--ag-min-words", type=int, default=AGNEWS_MIN_WORDS,
                   help=f"AG News: minimum words per article (default: {AGNEWS_MIN_WORDS})")
    p.add_argument("--ag-max-words", type=int, default=AGNEWS_MAX_WORDS,
                   help=f"AG News: truncate articles beyond this word count (default: {AGNEWS_MAX_WORDS})")
    p.add_argument("--ag-categories", type=int, nargs="*", default=None,
                   help="AG News category ids to keep: 0=World 1=Sports 2=Business 3=Sci/Tech (default: all)")

    # Dataset size
    p.add_argument("--num-samples", "-n", type=int, default=DEFAULT_NUM_SAMPLES,
                   help=f"Target number of paired samples (default: {DEFAULT_NUM_SAMPLES})")

    # Output
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: outputs/dataset)")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--light-constraint", action="store_true", default=True,
                   help="Light topic constraint for guided continuation (default: on)")
    p.add_argument("--content-align", action="store_true",
                   help="Enable full two-step content-aligned prompt (overrides --light-constraint)")
    p.add_argument("--free-only", action="store_true",
                   help="Only generate free (unconstrained) continuations, no guided")
    p.add_argument("--no-free", action="store_true",
                   help="Skip free continuation, only generate guided")
    p.add_argument("--no-content-align", action="store_true",
                   help="(Deprecated) kept for compatibility")
    p.add_argument("--workers", "-w", type=int, default=5,
                   help="Number of parallel workers for API calls (default: 5)")
    p.add_argument("--show-config", action="store_true",
                   help="Show resolved config and exit")
    return p.parse_args()


# ============================================================
# Step 1 — Load corpus
# ============================================================
def load_corpus(args) -> list:
    """Load documents from the configured corpus source."""
    loader = CorpusLoader()
    documents = []

    if args.corpus_jsonl:
        logger.info(f"Loading corpus from local JSONL: {args.corpus_jsonl}")
        for doc in loader.load_from_jsonl(
            filepath=args.corpus_jsonl,
            domain="custom",
            language=args.language,
            max_docs=args.max_docs,
        ):
            documents.append(doc)

    elif args.corpus == "ag_news":
        logger.info(
            f"Loading AG News (min_words={args.ag_min_words}, "
            f"max_words={args.ag_max_words}) ..."
        )
        for doc in loader.load_ag_news(
            max_docs=args.max_docs,
            min_words=args.ag_min_words,
            max_words=args.ag_max_words,
            categories=args.ag_categories,
        ):
            documents.append(doc)

    elif args.corpus == "cnn_dailymail":
        logger.info(
            f"Loading CNN/DailyMail (min_words={CNN_MIN_WORDS}, "
            f"max_words={CNN_MAX_WORDS}) ..."
        )
        for doc in loader.load_cnn_dailymail(
            max_docs=args.max_docs,
            min_words=CNN_MIN_WORDS,
            max_words=CNN_MAX_WORDS,
        ):
            documents.append(doc)

    else:
        # Wikipedia + optional WikiHow
        logger.info(f"Loading Wikipedia ({args.language}), max {args.max_docs} docs ...")
        for doc in loader.load_wikipedia(
            language=args.language,
            max_docs=args.max_docs,
        ):
            documents.append(doc)

        if args.max_wikihow_docs > 0:
            logger.info(f"Loading WikiHow, max {args.max_wikihow_docs} docs ...")
            try:
                for doc in loader.load_wikihow(
                    language=args.language,
                    max_docs=args.max_wikihow_docs,
                ):
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"WikiHow loading failed (non-fatal): {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


# ============================================================
# Step 2 — Split into prefix / continuation pairs
# ============================================================
def extract_pairs(documents, args):
    """Use PrefixSplitter to produce prefix-continuation pairs."""
    # Choose token targets based on corpus
    if args.corpus == "cnn_dailymail" and not args.corpus_jsonl:
        prefix_tokens = CNN_PREFIX_TOKENS
        cont_tokens = CNN_CONTINUATION_TOKENS
    elif args.corpus == "ag_news" and not args.corpus_jsonl:
        prefix_tokens = AGNEWS_PREFIX_TOKENS
        cont_tokens = AGNEWS_CONTINUATION_TOKENS
    else:
        prefix_tokens = WIKI_PREFIX_TOKENS
        cont_tokens = WIKI_CONTINUATION_TOKENS

    logger.info(
        f"Splitting with prefix={prefix_tokens} tokens, "
        f"continuation={cont_tokens} tokens (±{int(TOKEN_TOLERANCE*100)}%)"
    )

    splitter = PrefixSplitter(
        target_prefix_tokens=prefix_tokens,
        target_continuation_tokens=cont_tokens,
        tolerance=TOKEN_TOLERANCE,
    )

    pairs = list(splitter.process_documents(documents, max_pairs_per_doc=1))
    logger.info(f"Extracted {len(pairs)} prefix-continuation pairs")

    random.seed(args.seed)
    random.shuffle(pairs)

    if len(pairs) > args.num_samples:
        pairs = pairs[: args.num_samples]
        logger.info(f"Trimmed to {len(pairs)} pairs (target: {args.num_samples})")

    return pairs


# ============================================================
# Step 3 — Generate AI continuations via two-step pipeline
# ============================================================
def generate_ai_continuations(pairs, args):
    """
    Generate AI continuations with optional dual mode:
      - **Guided** (default): light topic+entity constraint from human text
      - **Free**: unconstrained "just continue" prompt
      - **Dual** (default): both guided and free

    When --content-align is set, full two-step content-aligned mode is used
    instead of light constraint.
    """
    from tqdm import tqdm

    generator = create_generator(
        backend=args.backend,
        model_path=args.model_path,
        model_name=args.model_name,
        api_key=args.api_key,
    )
    logger.info(f"Using backend model: {generator.model_name}")

    # Decide generation mode
    do_free = not args.no_free
    do_guided = not args.free_only

    prompt_builder = None
    if do_guided:
        if args.content_align:
            prompt_builder = PromptBuilder(generator=generator, temperature=0.3)
            logger.info("Content-aligned mode: full two-step constraint extraction")
        else:
            prompt_builder = PromptBuilder(generator=generator, temperature=0.3, light=True)
            logger.info("Light-constraint mode: topic + key entities only")

    cont_gen = ContinuationGenerator(
        generator=generator,
        model_name=generator.model_name,
        model_type="chat",
        prompt_builder=prompt_builder,
    )

    num_workers = max(1, args.workers)
    mode_desc = []
    if do_guided:
        mode_desc.append("guided")
    if do_free:
        mode_desc.append("free")
    logger.info(f"Generation mode: {' + '.join(mode_desc)}, workers: {num_workers}")

    def _process_one(pair):
        """Process a single pair — generate guided and/or free continuations."""
        result = None

        # Guided continuation (with constraints)
        if do_guided:
            result = cont_gen.generate(
                prefix=pair.prefix,
                human_continuation=pair.continuation,
                sample_id=pair.sample_id,
                doc_id=pair.doc_id,
                temperature=args.temperature,
            )

        # Free continuation (unconstrained)
        if do_free and result is not None:
            free_text = cont_gen.generate_free(
                prefix=pair.prefix,
                human_continuation=pair.continuation,
                sample_id=pair.sample_id,
                doc_id=pair.doc_id,
                temperature=args.temperature,
            )
            if free_text:
                result.ai_continuation_free = free_text
        elif do_free and not do_guided:
            # Free-only mode: generate free and wrap in a result
            free_text = cont_gen.generate_free(
                prefix=pair.prefix,
                human_continuation=pair.continuation,
                sample_id=pair.sample_id,
                doc_id=pair.doc_id,
                temperature=args.temperature,
            )
            if free_text:
                from src.data.continuation_gen import GeneratedContinuation
                result = GeneratedContinuation(
                    sample_id=pair.sample_id,
                    doc_id=pair.doc_id,
                    prefix=pair.prefix,
                    human_continuation=pair.continuation,
                    ai_continuation=free_text,
                    model_name=generator.model_name,
                    model_type="chat",
                    generation_params={"temperature": args.temperature, "content_aligned": False},
                    ai_continuation_free=free_text,
                )

        return result

    results = []
    failed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        future_to_pair = {pool.submit(_process_one, p): p for p in pairs}
        pbar = tqdm(total=len(pairs), desc="Generating AI continuations", unit="sample")
        for future in as_completed(future_to_pair):
            try:
                result = future.result()
            except Exception as exc:
                logger.warning(f"Worker exception: {exc}")
                result = None
            if result:
                results.append(result)
            else:
                failed += 1
            pbar.set_postfix(ok=len(results), fail=failed)
            pbar.update(1)
        pbar.close()

    logger.info(f"Generated {len(results)}/{len(pairs)} AI continuations")
    n_free = sum(1 for r in results if r.ai_continuation_free)
    logger.info(f"  Guided: {len(results)}, Free: {n_free}")
    return results


# ============================================================
# Step 4 — Assemble & save PairedDataset
# ============================================================
def assemble_and_save(pairs, ai_results, args):
    """Merge human & AI continuations and save as JSON."""
    ai_by_id = {r.sample_id: r for r in ai_results}

    samples = []
    for pair in pairs:
        ai = ai_by_id.get(pair.sample_id)
        if ai is None:
            continue

        merged_meta = dict(pair.metadata)
        if ai.content_constraint:
            merged_meta["content_constraint"] = ai.content_constraint
        merged_meta["generation_params"] = ai.generation_params

        sample = PairedSample(
            sample_id=pair.sample_id,
            doc_id=pair.doc_id,
            language=pair.language,
            domain=pair.domain,
            prefix=pair.prefix,
            human_continuation=pair.continuation,
            ai_continuation_chat=ai.ai_continuation,
            ai_continuation_free=ai.ai_continuation_free,
            metadata=merged_meta,
        )
        samples.append(sample)

    output_dir = Path(args.output_dir) if args.output_dir else get_dataset_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Include API model name in filename for traceability
    api_model = ai_results[0].model_name if ai_results else "unknown"
    api_model_id = get_model_id(api_model)
    out_path = output_dir / f"paired_dataset_{api_model_id}.json"

    out_data = {
        "metadata": {
            "model": api_model,
            "n_samples": len(samples),
            "language": args.language,
        },
        "samples": [s.to_dict() for s in samples],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(samples)} samples to {out_path}")

    return samples


# ============================================================
# Step 5 — Print statistics
# ============================================================
def print_stats(samples):
    n = len(samples)
    h_lens = [len(s.human_continuation.split()) for s in samples]
    a_lens = [len(s.ai_continuation_chat.split()) for s in samples if s.ai_continuation_chat]
    f_lens = [len(s.ai_continuation_free.split()) for s in samples if s.ai_continuation_free]
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"  Total samples      : {n}")
    print(f"  Human words        : mean={sum(h_lens)/n:.1f}  min={min(h_lens)}  max={max(h_lens)}")
    if a_lens:
        print(f"  AI guided words    : mean={sum(a_lens)/len(a_lens):.1f}  min={min(a_lens)}  max={max(a_lens)}  (n={len(a_lens)})")
    if f_lens:
        print(f"  AI free words      : mean={sum(f_lens)/len(f_lens):.1f}  min={min(f_lens)}  max={max(f_lens)}  (n={len(f_lens)})")
    print("=" * 60)


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    if args.show_config:
        from src.generation.backends import get_config_from_env
        cfg = get_config_from_env()
        print("Resolved configuration:")
        for k, v in cfg.items():
            if "key" in k.lower() and v:
                v = v[:8] + "..."
            print(f"  {k}: {v}")
        return

    print("\n" + "=" * 60)
    print("Building Paired Dataset from Real Corpora")
    print("=" * 60)

    # 1. Load corpus
    documents = load_corpus(args)
    if not documents:
        logger.error("No documents loaded. Check data source settings.")
        sys.exit(1)

    # 2. Extract pairs
    pairs = extract_pairs(documents, args)
    if not pairs:
        logger.error("No prefix-continuation pairs extracted.")
        sys.exit(1)

    # 3. Generate AI continuations
    ai_results = generate_ai_continuations(pairs, args)
    if not ai_results:
        logger.error("No AI continuations generated.")
        sys.exit(1)

    # 4. Save
    samples = assemble_and_save(pairs, ai_results, args)

    # 5. Stats
    print_stats(samples)

    print("\n" + "=" * 60)
    print("✅ Dataset construction complete!")
    print("=" * 60)
    print("\nNext step:")
    print("  python scripts/steering_pipeline.py")


if __name__ == "__main__":
    main()
