"""Generate AI continuations for prefixes."""

import logging
import re
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass 
class GeneratedContinuation:
    """AI-generated continuation for a prefix."""
    sample_id: str
    doc_id: str
    prefix: str
    human_continuation: str
    ai_continuation: str
    model_name: str
    model_type: str  # 'base' or 'chat'
    generation_params: dict = field(default_factory=dict)
    content_constraint: Optional[dict] = None
    ai_continuation_free: Optional[str] = None
    
    @property
    def ai_tokens(self) -> int:
        """Approximate token count."""
        return len(self.ai_continuation.split())


class ContinuationGenerator:
    """Generate AI continuations matching human continuation length.
    
    Supports two modes:
      - **Simple** (prompt_builder=None): generic "continue the text" prompt.
      - **Content-aligned** (prompt_builder provided): two-step pipeline where
        the human continuation is first analysed to extract topic / entities /
        direction, and the resulting constraints are injected into the
        continuation prompt so the AI writes about the *same content*.
    """
    
    # Fallback system prompt when no PromptBuilder is provided
    CHAT_SYSTEM_PROMPT = """Continue the given text directly. Write naturally. Do not add titles, bullet points, or meta-commentary."""
    
    CHAT_USER_TEMPLATE = """{prefix} /no_think"""
    
    def __init__(
        self,
        generator,
        model_name: str,
        model_type: str = "chat",
        target_length_tolerance: float = 0.2,
        max_retries: int = 3,
        prompt_builder=None,
    ):
        """
        Args:
            generator: Backend generator (from src.generation)
            model_name: Name of the model
            model_type: 'base' or 'chat'
            target_length_tolerance: How close AI length should match human
            max_retries: Max attempts to generate acceptable continuation
            prompt_builder: Optional PromptBuilder for content-aligned generation
        """
        self.generator = generator
        self.model_name = model_name
        self.model_type = model_type
        self.target_length_tolerance = target_length_tolerance
        self.max_retries = max_retries
        self.prompt_builder = prompt_builder
    
    def generate(
        self,
        prefix: str,
        human_continuation: str,
        sample_id: str,
        doc_id: str,
        temperature: float = 0.7,
        **kwargs,
    ) -> Optional[GeneratedContinuation]:
        """
        Generate AI continuation matching human continuation length.
        
        If a ``prompt_builder`` was provided at init time, the generation
        uses a two-step content-aligned pipeline:
          1. Analyse human continuation → ContentConstraint
          2. Generate with constrained prompt
        Otherwise falls back to the simple generic prompt.
        """
        target_tokens = self._estimate_tokens(human_continuation)
        min_tokens = int(target_tokens * (1 - self.target_length_tolerance))
        max_tokens = int(target_tokens * (1 + self.target_length_tolerance))

        # --- Step 1: build prompt (content-aligned or simple) ---
        constraint = None
        if self.prompt_builder is not None:
            try:
                constraint = self.prompt_builder.analyse_continuation(
                    prefix=prefix,
                    human_continuation=human_continuation,
                )
                logger.debug(f"[{sample_id}] Extracted constraint: topic={constraint.topic!r}")
            except Exception as e:
                logger.warning(f"[{sample_id}] Prompt analysis failed, falling back: {e}")
                constraint = None

        # --- Step 2: generate continuation ---
        for attempt in range(self.max_retries):
            try:
                if self.model_type == "base":
                    ai_text = self._generate_base(
                        prefix=prefix,
                        max_tokens=max_tokens + 20,
                        temperature=temperature,
                        **kwargs,
                    )
                else:
                    human_words = len(human_continuation.split())
                    ai_text = self._generate_chat(
                        prefix=prefix,
                        max_tokens=max_tokens + 20,
                        temperature=temperature,
                        constraint=constraint,
                        target_words=human_words,
                        **kwargs,
                    )
                
                ai_text = self._clean_continuation(ai_text)
                ai_text = self._truncate_to_length(ai_text, target_tokens, max_tokens)
                
                ai_tokens = self._estimate_tokens(ai_text)
                if ai_tokens >= min_tokens:
                    return GeneratedContinuation(
                        sample_id=sample_id,
                        doc_id=doc_id,
                        prefix=prefix,
                        human_continuation=human_continuation,
                        ai_continuation=ai_text,
                        model_name=self.model_name,
                        model_type=self.model_type,
                        generation_params={
                            "temperature": temperature,
                            "target_tokens": target_tokens,
                            "actual_tokens": ai_tokens,
                            "attempt": attempt + 1,
                            "content_aligned": constraint is not None,
                        },
                        content_constraint=constraint.to_dict() if constraint else None,
                    )
                
                logger.debug(f"Attempt {attempt + 1}: Generated too short ({ai_tokens} < {min_tokens})")
                
            except Exception as e:
                logger.warning(f"Generation failed on attempt {attempt + 1}: {e}")
        
        logger.warning(f"Failed to generate valid continuation for {sample_id}")
        return None
    
    def _generate_base(
        self,
        prefix: str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> str:
        """Generate with base model (direct continuation)."""
        response = self.generator.generate(
            prompt=prefix,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        return response
    
    def _generate_chat(
        self,
        prefix: str,
        max_tokens: int,
        temperature: float,
        constraint=None,
        target_words: int = 150,
        **kwargs,
    ) -> str:
        """Generate with chat model, optionally using content constraints."""
        if constraint is not None and self.prompt_builder is not None:
            system_prompt, user_prompt = self.prompt_builder.build_continuation_prompt(
                prefix=prefix,
                constraint=constraint,
                target_words=target_words,
            )
        else:
            system_prompt = self.CHAT_SYSTEM_PROMPT
            user_prompt = self.CHAT_USER_TEMPLATE.format(prefix=prefix)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Use proper chat() if available, otherwise fall back to generate()
        if hasattr(self.generator, "chat"):
            response = self.generator.chat(
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            full_prompt = f"[System]\n{system_prompt}\n\n[User]\n{user_prompt}"
            response = self.generator.generate(
                prompt=full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        return response
    
    def _clean_continuation(self, text: str) -> str:
        """Clean AI continuation to remove artifacts."""
        if not text:
            return ""
        
        # Strip Qwen3-style <think>...</think> reasoning blocks
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        
        patterns_to_remove = [
            r'^(Sure|Certainly|Here\'s|I\'ll continue)[^.]*[.:]\s*',
            r'^(Continuing|To continue)[^.]*[.:]\s*',
            r'^\*\*[^*]+\*\*\s*',  # Bold headers
            r'^#+\s+.*\n',  # Markdown headers
            r'^\s*[-*]\s+',  # Bullet points at start
            r'\n\s*[-*]\s+.*$',  # Bullet points
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text
    
    def _truncate_to_length(self, text: str, target_tokens: int, max_tokens: int) -> str:
        """Truncate text to target length at sentence boundary."""
        words = text.split()
        
        if len(words) * 1.3 <= max_tokens:
            return text
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        token_count = 0
        
        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if token_count + sent_tokens > max_tokens:
                if result:
                    break
            result.append(sent)
            token_count += sent_tokens
            if token_count >= target_tokens:
                break
        
        return " ".join(result)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return int(len(text.split()) * 1.3)
    
    def generate_free(
        self,
        prefix: str,
        human_continuation: str,
        sample_id: str,
        doc_id: str,
        temperature: float = 0.7,
        **kwargs,
    ) -> Optional[str]:
        """Generate an unconstrained (free) continuation. Returns text only."""
        target_tokens = self._estimate_tokens(human_continuation)
        min_tokens = int(target_tokens * (1 - self.target_length_tolerance))
        max_tokens = int(target_tokens * (1 + self.target_length_tolerance))
        human_words = len(human_continuation.split())

        for attempt in range(self.max_retries):
            try:
                system_prompt = self.CHAT_SYSTEM_PROMPT
                user_prompt = self.CHAT_USER_TEMPLATE.format(prefix=prefix)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                if hasattr(self.generator, "chat"):
                    ai_text = self.generator.chat(
                        messages=messages,
                        max_new_tokens=max_tokens + 20,
                        temperature=temperature,
                    )
                else:
                    full_prompt = f"[System]\n{system_prompt}\n\n[User]\n{user_prompt}"
                    ai_text = self.generator.generate(
                        prompt=full_prompt,
                        max_new_tokens=max_tokens + 20,
                        temperature=temperature,
                    )
                ai_text = self._clean_continuation(ai_text)
                ai_text = self._truncate_to_length(ai_text, target_tokens, max_tokens)
                if self._estimate_tokens(ai_text) >= min_tokens:
                    return ai_text
            except Exception as e:
                logger.warning(f"[{sample_id}] Free generation attempt {attempt+1} failed: {e}")
        return None

    def generate_batch(
        self,
        pairs: Iterator,
        batch_size: int = 10,
        **kwargs,
    ) -> Iterator[GeneratedContinuation]:
        """
        Generate continuations for multiple prefix-continuation pairs.
        
        Args:
            pairs: Iterator of PrefixContinuationPair objects
            batch_size: Logging frequency
            
        Yields:
            GeneratedContinuation objects
        """
        count = 0
        success = 0
        
        for pair in pairs:
            result = self.generate(
                prefix=pair.prefix,
                human_continuation=pair.continuation,
                sample_id=pair.sample_id,
                doc_id=pair.doc_id,
                **kwargs,
            )
            
            count += 1
            if result:
                success += 1
                yield result
            
            if count % batch_size == 0:
                logger.info(f"Generated {success}/{count} continuations")
        
        logger.info(f"Finished: {success}/{count} successful generations")
