"""26 — byte-level tokenizer and chat template.

The real engine uses HuggingFace ``tokenizers`` plus a `hanashi` chat encoding.
For the study demo a byte tokenizer is enough: ids 0..255 are UTF-8 bytes,
256 is EOS, 257 is the DFlash mask/pad token.  It keeps the code small while
preserving the important boundary: **text -> token ids -> model -> text**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


EOS_TOKEN_ID = 256
MASK_TOKEN_ID = 257
VOCAB_SIZE = 258


@dataclass
class ChatMessage:
    role: str
    content: str

    @classmethod
    def system(cls, text: str) -> "ChatMessage":
        return cls("system", text)

    @classmethod
    def user(cls, text: str) -> "ChatMessage":
        return cls("user", text)

    @classmethod
    def assistant(cls, text: str) -> "ChatMessage":
        return cls("assistant", text)


class ByteTokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        if vocab_size < 258:
            raise ValueError("byte tokenizer needs at least 258 ids")
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_special_tokens:
            ids.append(EOS_TOKEN_ID)
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        raw = bytes(int(i) for i in ids if 0 <= int(i) < 256)
        return raw.decode("utf-8", errors="replace")

    def apply_chat_template(self, messages: Sequence[ChatMessage], add_generation_prompt: bool = True) -> str:
        parts: list[str] = []
        for message in messages:
            parts.append(f"<|{message.role}|>\n{message.content}\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)

    def encode_chat(self, messages: Sequence[ChatMessage]) -> list[int]:
        return self.encode(self.apply_chat_template(messages))
