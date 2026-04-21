"""Corpus loading and cleaning for CNN/DailyMail (primary), AG News, and Wikipedia."""

import re
import logging
from pathlib import Path
from typing import Iterator, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single document from corpus."""
    doc_id: str
    title: str
    text: str
    domain: str  # 'wikipedia', 'wikihow', or 'ag_news'
    language: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CorpusLoader:
    """Load and clean corpus from various sources."""
    
    # Patterns to remove
    CLEANUP_PATTERNS = [
        r'\[\d+\]',  # Citation markers [1], [2]
        r'\[citation needed\]',
        r'\[edit\]',
        r'<ref[^>]*>.*?</ref>',  # XML refs
        r'{{[^}]*}}',  # Wiki templates
        r'\[\[File:[^\]]*\]\]',  # File links
        r'\[\[Image:[^\]]*\]\]',  # Image links
    ]
    
    # Patterns indicating non-prose content
    SKIP_PATTERNS = [
        r'^[=]+.*[=]+$',  # Headers
        r'^\*\s',  # Bullet points
        r'^\d+\.\s',  # Numbered lists  
        r'^\|',  # Table rows
        r'^!',  # Table headers
        r'^See also',
        r'^References',
        r'^External links',
        r'^Notes',
        r'^Bibliography',
    ]
    
    # AG News source attribution patterns to strip
    # Typical format: "Headline (Source) Source - body text"
    # We remove the "(Source) Source - " part, keeping the headline.
    AG_NEWS_SOURCE_RE = re.compile(
        r'\s*'
        r'\([^)]{2,40}\)'                # (Reuters), (AP), (Forbes.com)
        r'\s*'
        r'[A-Za-z][A-Za-z0-9\s&.,/\']+?' # repeated source name
        r'\s*[-–—]\s*',                   # dash separator
    )

    AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    # CNN/DailyMail: leading byline patterns
    CNN_BYLINE_RE = re.compile(
        r"^(?:"
        r"[A-Z][A-Za-z .'-]+,\s*[A-Za-z .'-]+\s*\([A-Za-z]+\)\s*[-–—]\s*"  # CITY, Country (CNN) --
        r"|[A-Z][A-Za-z .'-]+\s*\([A-Za-z]+\)\s*[-–—]\s*"                   # CITY (Reuters) --
        r"|By\s+[A-Za-z .'-]+\s*(?:for\s+[A-Za-z .'-]+)?\s*"                # By Author for DailyMail
        r"|Editor'?s?\s+note:.*?[.]\s*"                                       # Editor's note: ...
        r")",
        re.IGNORECASE,
    )

    def __init__(self, cache_dir: Optional[str] = None, language: str = "en"):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            from src.config import get_raw_dir
            self.cache_dir = get_raw_dir(language)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_re = [re.compile(p, re.IGNORECASE) for p in self.CLEANUP_PATTERNS]
        self._skip_re = [re.compile(p, re.MULTILINE) for p in self.SKIP_PATTERNS]
    
    def load_wikipedia(
        self, 
        language: str = "en",
        max_docs: Optional[int] = None,
        min_length: int = 500,
    ) -> Iterator[Document]:
        """Load Wikipedia articles from HuggingFace datasets.

        Articles are cached to ``data/raw/{language}/wikipedia_{N}.jsonl`` on
        first load so that downstream consumers (held-out prompt building,
        secondary analysis) can reuse the same documents without re-streaming
        from HuggingFace.
        """
        import json as _json
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        cache_name = f"wikipedia_{max_docs or 'all'}.jsonl"
        cache_path = self.cache_dir / cache_name
        if cache_path.is_file():
            logger.info(f"Loading cached Wikipedia from {cache_path}")
            yield from self.load_from_jsonl(
                str(cache_path), domain="wikipedia", language=language,
                max_docs=max_docs, min_length=min_length,
            )
            return

        logger.info(f"Loading Wikipedia ({language})...")
        config = f"20231101.{language}"
        dataset = load_dataset(
            "wikimedia/wikipedia", config, split="train", streaming=True
        )
        
        count = 0
        cache_file = open(cache_path, "w", encoding="utf-8") if cache_path.parent.is_dir() else None
        try:
            for item in dataset:
                if max_docs and count >= max_docs:
                    break

                text = self._clean_text(item.get("text", ""))
                if len(text) < min_length:
                    continue

                doc = Document(
                    doc_id=f"wiki_{language}_{item.get('id', count)}",
                    title=item.get("title", ""),
                    text=text,
                    domain="wikipedia",
                    language=language,
                    metadata={"url": item.get("url", "")},
                )

                if cache_file:
                    cache_file.write(_json.dumps({
                        "doc_id": doc.doc_id, "title": doc.title,
                        "text": doc.text, "domain": doc.domain,
                        "language": doc.language, "metadata": doc.metadata,
                    }, ensure_ascii=False) + "\n")

                count += 1
                if count % 1000 == 0:
                    logger.info(f"Loaded {count} Wikipedia documents")

                yield doc
        finally:
            if cache_file:
                cache_file.close()
        
        logger.info(f"Finished loading {count} Wikipedia documents")
    
    def load_wikihow(
        self,
        language: str = "en",
        max_docs: Optional[int] = None,
        min_length: int = 500,
    ) -> Iterator[Document]:
        """Load WikiHow articles from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        logger.info(f"Loading WikiHow ({language})...")
        
        # WikiHow dataset - use the "all" config for full dataset
        dataset = load_dataset("wikihow", "all", split="train", streaming=True)
        
        count = 0
        for item in dataset:
            if max_docs and count >= max_docs:
                break
            
            # WikiHow has structured text - combine headline and text
            text_parts = []
            if item.get("title"):
                # Skip the title as it's often "How to..."
                pass
            if item.get("text"):
                text_parts.append(item["text"])
            
            text = self._clean_text(" ".join(text_parts))
            if len(text) < min_length:
                continue
            
            doc = Document(
                doc_id=f"wikihow_{language}_{count}",
                title=item.get("title", ""),
                text=text,
                domain="wikihow",
                language=language,
                metadata={},
            )
            
            count += 1
            if count % 1000 == 0:
                logger.info(f"Loaded {count} WikiHow documents")
            
            yield doc
        
        logger.info(f"Finished loading {count} WikiHow documents")
    
    def load_ag_news(
        self,
        max_docs: Optional[int] = None,
        min_words: int = 50,
        max_words: int = 120,
        categories: Optional[List[int]] = None,
    ) -> Iterator[Document]:
        """Load AG News articles from HuggingFace (fancyzhx/ag_news).

        Args:
            max_docs: Maximum number of documents to yield.
            min_words: Minimum word count after cleaning (skip shorter articles).
            max_words: Truncate articles exceeding this word count at a sentence
                       boundary so that all documents have comparable length.
            categories: Optional list of label ids to keep
                        (0=World, 1=Sports, 2=Business, 3=Sci/Tech).
                        ``None`` keeps all categories.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        logger.info(
            f"Loading AG News (min_words={min_words}, max_words={max_words}, "
            f"categories={categories or 'all'}) ..."
        )

        dataset = load_dataset("fancyzhx/ag_news", split="train", streaming=True)

        count = 0
        scanned = 0
        for item in dataset:
            if max_docs and count >= max_docs:
                break

            label = item.get("label", -1)
            if categories is not None and label not in categories:
                continue

            text = self._clean_ag_news(item.get("text", ""))
            words = text.split()
            scanned += 1

            if len(words) < min_words:
                continue

            # Truncate at sentence boundary if exceeding max_words
            if len(words) > max_words:
                text = self._truncate_at_sentence(text, max_words)
                if not text:
                    continue

            category = self.AG_NEWS_LABELS.get(label, "Unknown")
            doc = Document(
                doc_id=f"agnews_{count}",
                title="",
                text=text,
                domain=f"ag_news/{category}",
                language="en",
                metadata={"category": category, "label": label},
            )

            count += 1
            if count % 2000 == 0:
                logger.info(f"Loaded {count} AG News articles (scanned {scanned})")

            yield doc

        logger.info(
            f"Finished loading {count} AG News articles "
            f"(scanned {scanned}, filtered {scanned - count})"
        )

    def load_cnn_dailymail(
        self,
        max_docs: Optional[int] = None,
        min_words: int = 200,
        max_words: int = 600,
    ) -> Iterator[Document]:
        """Load CNN/DailyMail articles from HuggingFace (cnn_dailymail 3.0.0).

        These are full-length news articles (avg ~700 words), much longer
        than AG News, which makes them ideal for studying style differences
        in continuations.

        Args:
            max_docs: Maximum number of documents to yield.
            min_words: Minimum word count after cleaning.
            max_words: Truncate articles beyond this at a sentence boundary.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        logger.info(
            f"Loading CNN/DailyMail (min_words={min_words}, "
            f"max_words={max_words}) ..."
        )

        dataset = load_dataset(
            "cnn_dailymail", "3.0.0", split="train", streaming=True,
        )

        count = 0
        scanned = 0
        for item in dataset:
            if max_docs and count >= max_docs:
                break

            text = self._clean_cnn_dailymail(item.get("article", ""))
            words = text.split()
            scanned += 1

            if len(words) < min_words:
                continue

            if len(words) > max_words:
                text = self._truncate_at_sentence(text, max_words)
                if not text or len(text.split()) < min_words:
                    continue

            doc = Document(
                doc_id=f"cnndm_{count}",
                title="",
                text=text,
                domain="cnn_dailymail",
                language="en",
                metadata={"id": item.get("id", "")},
            )

            count += 1
            if count % 2000 == 0:
                logger.info(f"Loaded {count} CNN/DM articles (scanned {scanned})")

            yield doc

        logger.info(
            f"Finished loading {count} CNN/DM articles "
            f"(scanned {scanned}, filtered {scanned - count})"
        )

    def _clean_cnn_dailymail(self, text: str) -> str:
        """Clean CNN/DailyMail article text."""
        if not text:
            return ""
        # Remove byline / attribution header
        text = self.CNN_BYLINE_RE.sub("", text)
        # CNN/DM uses (CNN) style markers scattered through text
        text = re.sub(r"\(CNN\)\s*[-–—]?\s*", "", text)
        # Remove embedded photo/video captions
        text = re.sub(r"(?:Watch|See|Click)\s.*?(?:video|gallery|photo).*?\.", "", text, flags=re.IGNORECASE)
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_chinese_news(
        self,
        max_docs: Optional[int] = None,
        min_chars: int = 200,
        max_chars: int = 1500,
        balance_categories: bool = True,
        per_category_cap: Optional[int] = None,
    ) -> Iterator[Document]:
        """Load Chinese news articles from seamew/THUCNews on HuggingFace.

        THUCNews contains ~740K articles from Sina News across 14 categories
        (体育, 娱乐, 财经, 房产, 教育, 科技, 时尚, 时政, 游戏, 家居, 彩票,
        股票, 社会, 星座). The streaming order is **sorted by category**, so a
        naive read returns ~131K consecutive 体育 articles before any other
        topic appears.

        When ``balance_categories=True`` (default) we round-robin across
        categories with a per-category cap of ``ceil(max_docs / 14)`` (or the
        explicit ``per_category_cap``). This guarantees topical diversity in
        the cached corpus.

        Articles are cached to ``data/raw/zh/thucnews_{max_docs}.jsonl``.
        """
        import json as _json
        import math
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        cache_name = f"thucnews_{max_docs or 'all'}.jsonl"
        cache_path = self.cache_dir / cache_name
        if cache_path.is_file():
            logger.info(f"Loading cached THUCNews from {cache_path}")
            yield from self.load_from_jsonl(
                str(cache_path), domain="chinese_news", language="zh",
                max_docs=max_docs, min_length=min_chars,
            )
            return

        logger.info("Loading Chinese news corpus (seamew/THUCNews)...")

        try:
            dataset = load_dataset("dirtycomputer/THUCNews", split="train", streaming=True)
            source = "thucnews"
        except Exception as e:
            logger.warning(f"THUCNews unavailable ({e}), falling back to Chinese Wikipedia")
            yield from self.load_wikipedia(language="zh", max_docs=max_docs, min_length=min_chars)
            return

        # Per-category quota for diversification
        N_CATS = 14  # THUCNews has 14 categories
        if per_category_cap is None and balance_categories and max_docs:
            per_category_cap = max(2, math.ceil(max_docs / N_CATS))
        cat_counts: dict = {}

        count = 0
        scanned = 0
        cache_file = open(cache_path, "w", encoding="utf-8") if cache_path.parent.is_dir() else None
        try:
            for item in dataset:
                scanned += 1
                if max_docs and count >= max_docs:
                    break

                raw_text = item.get("text", "")
                # dirtycomputer/THUCNews format: "category\ttext"
                if "\t" in raw_text:
                    category, text = raw_text.split("\t", 1)
                else:
                    category, text = "", raw_text

                # Enforce per-category cap to ensure topical balance
                if balance_categories and per_category_cap is not None:
                    if cat_counts.get(category, 0) >= per_category_cap:
                        continue

                title = ""

                text = self._clean_chinese_text(text)

                if len(text) < min_chars:
                    continue

                if len(text) > max_chars:
                    text = self._truncate_at_sentence(text, max_chars, language="zh")

                doc = Document(
                    doc_id=f"zhnews_{count}",
                    title=str(title),
                    text=text,
                    domain="chinese_news",
                    language="zh",
                    metadata={"source": source, "category": category},
                )

                # Cache to disk
                if cache_file:
                    cache_file.write(_json.dumps({
                        "doc_id": doc.doc_id, "title": doc.title,
                        "text": doc.text, "domain": doc.domain,
                        "language": doc.language, "metadata": doc.metadata,
                    }, ensure_ascii=False) + "\n")

                count += 1
                cat_counts[category] = cat_counts.get(category, 0) + 1
                if count % 500 == 0:
                    logger.info(
                        f"Loaded {count} Chinese news articles (scanned {scanned}, "
                        f"categories so far: {len(cat_counts)})"
                    )

                yield doc
        finally:
            if cache_file:
                cache_file.close()

        logger.info(
            f"Finished loading {count} Chinese news articles "
            f"(scanned {scanned}, filtered {scanned - count})"
        )
        if cache_path.is_file():
            logger.info(f"Cached {count} articles to {cache_path}")

    def _clean_chinese_text(self, text: str) -> str:
        """Clean Chinese text: remove markup, normalize whitespace."""
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove 【】 and 《》 markup content
        text = re.sub(r"【[^】]*】", "", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove excess whitespace but keep Chinese characters together
        text = re.sub(r"[\t\r]+", "", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _clean_ag_news(self, text: str) -> str:
        """Strip source attributions and normalise whitespace for AG News."""
        if not text:
            return ""
        # Replace escaped newlines / backslash artifacts
        text = text.replace("\\", " ")
        # Remove "(Source) Source — " attribution (e.g. "(Reuters) Reuters - ")
        text = self.AG_NEWS_SOURCE_RE.sub(" ", text)
        # Also remove bare leading attribution without parenthetical
        # e.g. "Reuters - body" or "AP - body"
        text = re.sub(r"^[A-Z][A-Za-z.]{1,30}\s*[-–—]\s*", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _truncate_at_sentence(self, text: str, max_words: int, language: str = "en") -> str:
        """Truncate *text* to at most *max_words* at a sentence boundary."""
        if language in ("zh", "chinese"):
            sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", text)
        result_words = 0
        kept: List[str] = []
        for sent in sentences:
            sw = len(sent.split()) if language not in ("zh", "chinese") else len(sent)
            if result_words + sw > max_words and kept:
                break
            kept.append(sent)
            result_words += sw
        joiner = "" if language in ("zh", "chinese") else " "
        return joiner.join(kept) if kept else ""

    def load_from_jsonl(
        self,
        filepath: str,
        domain: str = "wikipedia",
        language: str = "en",
        max_docs: Optional[int] = None,
        min_length: int = 500,
    ) -> Iterator[Document]:
        """Load documents from local JSONL file."""
        import json
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading from {filepath}...")
        
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if max_docs and count >= max_docs:
                    break
                
                item = json.loads(line)
                text = item.get("text", "")
                if len(text) < min_length:
                    continue
                
                doc = Document(
                    doc_id=item.get("doc_id", item.get("id", f"{domain}_{language}_{count}")),
                    title=item.get("title", ""),
                    text=text,
                    domain=item.get("domain", domain),
                    language=item.get("language", language),
                    metadata=item.get("metadata", {}),
                )
                
                count += 1
                yield doc
        
        logger.info(f"Finished loading {count} documents from {filepath}")
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing wiki markup and normalizing."""
        if not text:
            return ""
        
        # Apply cleanup patterns
        for pattern in self._cleanup_re:
            text = pattern.sub("", text)
        
        # Split into lines and filter
        lines = text.split("\n")
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip non-prose lines
            skip = False
            for pattern in self._skip_re:
                if pattern.match(line):
                    skip = True
                    break
            
            if not skip:
                clean_lines.append(line)
        
        # Join and normalize whitespace
        text = " ".join(clean_lines)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def extract_paragraphs(
        self, 
        doc: Document, 
        min_sentences: int = 3,
        max_sentences: int = 15,
    ) -> List[str]:
        """Extract clean paragraphs from document."""
        # Split on double newlines or periods followed by capitals
        text = doc.text
        
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        if len(sentences) < min_sentences:
            return []
        
        # Return sentences that can form valid paragraphs
        paragraphs = []
        for i in range(0, len(sentences) - min_sentences + 1, min_sentences):
            end = min(i + max_sentences, len(sentences))
            para = " ".join(sentences[i:end])
            if len(para) >= 200:  # Minimum paragraph length
                paragraphs.append(para)
        
        return paragraphs
