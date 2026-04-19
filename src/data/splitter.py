"""Split documents into prefix-continuation pairs."""

import re
import logging
from typing import List, Tuple, Optional, Iterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PrefixContinuationPair:
    """A prefix-continuation pair extracted from a document."""
    sample_id: str
    doc_id: str
    prefix: str
    continuation: str
    prefix_tokens: int
    continuation_tokens: int
    domain: str
    language: str
    metadata: dict = field(default_factory=dict)


class PrefixSplitter:
    """Split text into prefix-continuation pairs with token-based windowing."""
    
    def __init__(
        self,
        tokenizer=None,
        target_prefix_tokens: int = 96,
        target_continuation_tokens: int = 160,
        tolerance: float = 0.2,
    ):
        """
        Initialize splitter.
        
        Args:
            tokenizer: HuggingFace tokenizer (will use tiktoken if None)
            target_prefix_tokens: Target number of tokens for prefix (~96)
            target_continuation_tokens: Target number of tokens for continuation (~160)
            tolerance: Allowed deviation from target (±20%)
        """
        self.tokenizer = tokenizer
        self.target_prefix_tokens = target_prefix_tokens
        self.target_continuation_tokens = target_continuation_tokens
        self.tolerance = tolerance
        
        self.min_prefix = int(target_prefix_tokens * (1 - tolerance))
        self.max_prefix = int(target_prefix_tokens * (1 + tolerance))
        self.min_continuation = int(target_continuation_tokens * (1 - tolerance))
        self.max_continuation = int(target_continuation_tokens * (1 + tolerance))
        
        self._tiktoken_enc = None
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text to get token count."""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        
        # Fallback to tiktoken for counting
        if self._tiktoken_enc is None:
            try:
                import tiktoken
                self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                # Very rough estimate: 1 token ≈ 4 chars for English
                return list(range(len(text) // 4))
        
        return self._tiktoken_enc.encode(text)
    
    def _detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text."""
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens)
        
        if self._tiktoken_enc is not None:
            return self._tiktoken_enc.decode(tokens)
        
        raise ValueError("No tokenizer available for detokenization")
    
    def split_text(
        self, 
        text: str,
        doc_id: str,
        domain: str,
        language: str,
        max_pairs: int = 1,
    ) -> List[PrefixContinuationPair]:
        """
        Split text into prefix-continuation pairs.
        
        Args:
            text: Input text to split
            doc_id: Document identifier
            domain: Source domain (wikipedia, wikihow)
            language: Language code
            max_pairs: Maximum number of pairs to extract per document
            
        Returns:
            List of PrefixContinuationPair objects
        """
        # Tokenize full text
        tokens = self._tokenize(text)
        total_tokens = len(tokens)
        
        min_required = self.min_prefix + self.min_continuation
        if total_tokens < min_required:
            logger.debug(f"Text too short: {total_tokens} < {min_required} tokens")
            return []
        
        pairs = []
        pair_idx = 0
        
        # Try to find good split points at sentence boundaries
        sentences = self._split_sentences(text, language=language)
        if not sentences:
            return []
        
        current_tokens = 0
        prefix_end_idx = 0
        
        # Find sentence boundary closest to target prefix length
        for i, sent in enumerate(sentences):
            sent_tokens = len(self._tokenize(sent))
            
            if current_tokens + sent_tokens >= self.min_prefix:
                if current_tokens + sent_tokens <= self.max_prefix:
                    prefix_end_idx = i + 1
                    break
                elif current_tokens >= self.min_prefix:
                    prefix_end_idx = i
                    break
                else:
                    prefix_end_idx = i + 1
                    break
            
            current_tokens += sent_tokens
        
        if prefix_end_idx == 0:
            prefix_end_idx = 1
        
        # Extract prefix
        prefix = " ".join(sentences[:prefix_end_idx])
        prefix_tokens = self._tokenize(prefix)
        
        # Extract continuation
        remaining_sentences = sentences[prefix_end_idx:]
        continuation_parts = []
        cont_token_count = 0
        
        for sent in remaining_sentences:
            sent_tokens = len(self._tokenize(sent))
            if cont_token_count + sent_tokens > self.max_continuation:
                if cont_token_count >= self.min_continuation:
                    break
            continuation_parts.append(sent)
            cont_token_count += sent_tokens
            if cont_token_count >= self.target_continuation_tokens:
                break
        
        if not continuation_parts or cont_token_count < self.min_continuation:
            logger.debug(f"Continuation too short: {cont_token_count} tokens")
            return []
        
        continuation = " ".join(continuation_parts)
        
        pair = PrefixContinuationPair(
            sample_id=f"{doc_id}_{pair_idx}",
            doc_id=doc_id,
            prefix=prefix,
            continuation=continuation,
            prefix_tokens=len(prefix_tokens),
            continuation_tokens=cont_token_count,
            domain=domain,
            language=language,
            metadata={
                "prefix_sentences": prefix_end_idx,
                "continuation_sentences": len(continuation_parts),
            },
        )
        pairs.append(pair)
        
        return pairs[:max_pairs]
    
    def _split_sentences(self, text: str, language: str = "en") -> List[str]:
        """Split text into sentences, handling language-specific punctuation."""
        if language in ("zh", "chinese"):
            # Chinese sentence boundaries: 。！？ plus English .!?
            sentences = re.split(r'(?<=[。！？.!?])\s*', text)
            min_len = 5  # Chinese chars are denser
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            min_len = 10
        
        sentences = [s.strip() for s in sentences if len(s.strip()) > min_len]
        
        return sentences
    
    def process_documents(
        self,
        documents: Iterator,
        max_pairs_per_doc: int = 1,
    ) -> Iterator[PrefixContinuationPair]:
        """
        Process multiple documents and yield prefix-continuation pairs.
        
        Args:
            documents: Iterator of Document objects
            max_pairs_per_doc: Maximum pairs per document
            
        Yields:
            PrefixContinuationPair objects
        """
        for doc in documents:
            pairs = self.split_text(
                text=doc.text,
                doc_id=doc.doc_id,
                domain=doc.domain,
                language=doc.language,
                max_pairs=max_pairs_per_doc,
            )
            yield from pairs
