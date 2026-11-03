"""29 — engine, language model, chat session.

Mirrors ``engine/mod.rs`` and ``engine/language_model/mod.rs``.

This is the public surface from the README:

    engine = Engine.new(EngineConfig())
    model  = engine.load_random(tiny_config())
    session = engine.chat(model, ChatConfig())
    replies = session.reply(messages, ChatReplyConfig())

For a real checkpoint, replace ``load_random`` with a loader that reads
``config.json`` + ``model.safetensors`` into a ``ParameterTree``.  The rest of
the graph is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from study_book.chapter_01_foundations.backend import CPUContext, DeviceCapabilities
from study_book.chapter_02_configuration.config import GenerationConfig, ModelConfig
from study_book.chapter_02_configuration.safetensors import ParameterTree, load_safetensors
from study_book.chapter_04_generation.sampling import Sampling
from study_book.chapter_04_generation.speculator import DFlashTfmSpeculator
from study_book.chapter_05_runtime.tokenizer import ByteTokenizer, ChatMessage
from study_book.chapter_05_runtime.decoder import Decoder
from study_book.chapter_05_runtime.stream import GenerationStream, StreamMetrics


@dataclass
class EngineConfig:
    allow_ollama_usage: bool = False
    allow_lmstudio_usage: bool = False
    device_name: str = "python-cpu"
    supports_mxu: bool = False


@dataclass
class ModelInfo:
    name: str
    path: Optional[Path] = None


@dataclass
class ChatConfig:
    generation: Optional[GenerationConfig] = None


@dataclass
class ChatReplyConfig:
    generation: Optional[GenerationConfig] = None


@dataclass
class ChatReply:
    message: ChatMessage
    stats: StreamMetrics


@dataclass
class ChatStreamChunk:
    token_id: int
    text: str
    finished: bool = False
    stats: Optional[StreamMetrics] = None


class LanguageModel:
    def __init__(
        self,
        config: ModelConfig,
        params: ParameterTree,
        context: CPUContext,
    ) -> None:
        self.config = config
        self.context = context
        self.decoder = Decoder(config.decoder_config, params.subtree("decoder"))
        self.tokenizer = ByteTokenizer(config.tokenizer_vocab_size)
        self.sampling = Sampling(config.decoder_config.vocab_size)
        self.generation_config = config.generation_config
        self.speculator: Optional[DFlashTfmSpeculator] = None
        if config.speculator_config is not None:
            self.speculator = DFlashTfmSpeculator(
                config.speculator_config,
                params.subtree("speculator"),
                self.decoder.embedding,
            )

    def max_context_length(self) -> Optional[int]:
        return self.decoder.max_context_length()


class ChatSession:
    def __init__(self, model: LanguageModel, config: Optional[ChatConfig] = None) -> None:
        self.model = model
        self.config = config or ChatConfig()

    def _generation_config(self, reply_config: Optional[ChatReplyConfig]) -> GenerationConfig:
        if reply_config is not None and reply_config.generation is not None:
            return reply_config.generation
        if self.config.generation is not None:
            return self.config.generation
        return self.model.generation_config

    def reply(
        self,
        messages: Sequence[ChatMessage],
        config: Optional[ChatReplyConfig] = None,
    ) -> list[ChatReply]:
        prompt = self.model.tokenizer.encode_chat(messages)
        stream = GenerationStream(self.model, prompt, self._generation_config(config))
        tokens = list(stream)
        text = self.model.tokenizer.decode(tokens)
        return [ChatReply(ChatMessage.assistant(text), stream.metrics)]

    def reply_with_stream(
        self,
        messages: Sequence[ChatMessage],
        config: Optional[ChatReplyConfig] = None,
    ) -> Iterator[ChatStreamChunk]:
        prompt = self.model.tokenizer.encode_chat(messages)
        stream = GenerationStream(self.model, prompt, self._generation_config(config))
        for token in stream:
            text = self.model.tokenizer.decode([token])
            yield ChatStreamChunk(token_id=token, text=text)
        yield ChatStreamChunk(token_id=-1, text="", finished=True, stats=stream.metrics)


class Engine:
    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        capabilities = DeviceCapabilities(
            device_name=self.config.device_name,
            supports_mxu=self.config.supports_mxu,
        )
        self.context = CPUContext(capabilities)
        self._models: dict[str, LanguageModel] = {}

    @classmethod
    def new(cls, config: Optional[EngineConfig] = None) -> "Engine":
        return cls(config)

    # -- model loading ----------------------------------------------------
    def load_random(self, config: ModelConfig, seed: int = 0, name: str = "tiny-random") -> ModelInfo:
        params = ParameterTree(flat={}, rng=np.random.default_rng(seed), allow_random=True)
        model = LanguageModel(config, params, self.context)
        self._models[name] = model
        return ModelInfo(name=name)

    def load_safetensors_model(
        self,
        config: ModelConfig,
        model_path: str | Path,
        name: str = "local",
    ) -> ModelInfo:
        path = Path(model_path)
        tensors = load_safetensors(path / "model.safetensors")
        params = ParameterTree(flat=tensors, allow_random=False)
        model = LanguageModel(config, params, self.context)
        self._models[name] = model
        return ModelInfo(name=name, path=path)

    def model(self, name: str) -> Optional[ModelInfo]:
        return ModelInfo(name=name) if name in self._models else None

    def language_model(self, info: ModelInfo) -> LanguageModel:
        return self._models[info.name]

    def chat(self, model: ModelInfo, config: Optional[ChatConfig] = None) -> ChatSession:
        return ChatSession(self.language_model(model), config)
