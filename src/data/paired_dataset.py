"""Paired continuation sample dataclass."""

from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class PairedSample:
    """A complete sample with human and AI continuations."""
    sample_id: str
    doc_id: str
    language: str
    domain: str
    prefix: str
    human_continuation: str
    ai_continuation_base: Optional[str] = None
    ai_continuation_chat: Optional[str] = None
    ai_continuation_free: Optional[str] = None
    prefix_len: int = 0
    human_len: int = 0
    ai_base_len: int = 0
    ai_chat_len: int = 0
    ai_free_len: int = 0
    split: str = "train"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.prefix_len == 0:
            self.prefix_len = len(self.prefix.split())
        if self.human_len == 0:
            self.human_len = len(self.human_continuation.split())
        if self.ai_base_len == 0 and self.ai_continuation_base:
            self.ai_base_len = len(self.ai_continuation_base.split())
        if self.ai_chat_len == 0 and self.ai_continuation_chat:
            self.ai_chat_len = len(self.ai_continuation_chat.split())
        if self.ai_free_len == 0 and self.ai_continuation_free:
            self.ai_free_len = len(self.ai_continuation_free.split())

    def is_complete(self) -> bool:
        return bool(self.human_continuation and
                    (self.ai_continuation_base or self.ai_continuation_chat))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PairedSample":
        return cls(**data)
