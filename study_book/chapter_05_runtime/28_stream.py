"""28 — generation stream and speculative verification.

Mirrors ``engine/language_model/stream/stream.rs``.

Two loops:

* normal decode: one token per target forward;
* speculative: DFlash + Weaver propose a tree, the target runs **one** forward
  over the whole tree with an ancestor mask, then we accept the longest matching
  path and sample one bonus token.

The cache is advanced with ``encode_accept(accepted_rows)``.  Rejected branches
are dropped by compacting the last KV chunk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import numpy as np

from study_book.chapter_02_configuration.config import GenerationConfig
from study_book.chapter_04_generation.trie import TrieNode
from study_book.chapter_04_generation.batch_topology import BatchTopology
from study_book.chapter_04_generation.sampling import PRng
from study_book.chapter_04_generation.weaver import ProposalNode


@dataclass
class StreamMetrics:
    num_prefill_forward_passes: int = 0
    num_decode_forward_passes: int = 0
    tokens_count_output: int = 0
    speculative_rounds: int = 0
    accepted_draft_tokens: int = 0
    first_token_seconds: Optional[float] = None
    total_seconds: float = 0.0

    @property
    def tokens_per_forward_pass(self) -> float:
        passes = self.num_prefill_forward_passes + self.num_decode_forward_passes
        return self.tokens_count_output / passes if passes else 0.0


class GenerationStream:
    def __init__(
        self,
        model,
        prompt_ids: Sequence[int],
        config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model = model
        self.prompt_ids = [int(t) for t in prompt_ids]
        self.config = config or model.generation_config
        self.sampling_config = self.config.sampling
        self.sampling = model.sampling
        self.decoder = model.decoder
        self.speculator = model.speculator
        self.prng = PRng(self.sampling_config.seed)
        self.state = self.decoder.create_empty_state(self.decoder.max_context_length(), None)
        self.metrics = StreamMetrics()
        self.generated: list[int] = []
        self._last_features: Optional[list[np.ndarray]] = None
        self._spec_state = (
            self.speculator.empty_state(self.decoder.max_context_length())
            if self.speculator is not None
            else None
        )
        self._spec_indices = (
            self.speculator.hidden_feature_layer_indices if self.speculator is not None else None
        )

    # -- helpers ----------------------------------------------------------
    def _sample(self, logits: np.ndarray) -> int:
        return int(self.sampling.sample_one(logits, self.sampling_config, prng=self.prng))

    def _capture_features(self, features: Optional[list[np.ndarray]], row: int) -> Optional[list[np.ndarray]]:
        if features is None:
            return None
        if row < 0:
            row = features[0].shape[0] + row
        return [np.asarray(f[row : row + 1], dtype=np.float32) for f in features]

    def _verify(
        self,
        nodes: Sequence[ProposalNode],
        logits: np.ndarray,
    ) -> tuple[list[int], list[int], int]:
        """Accept the longest child path that matches the sampled token."""
        children: dict[int, list[int]] = {i: list(n.children) for i, n in enumerate(nodes)}
        accepted_rows = [0]
        draft_tokens: list[int] = []
        current = 0
        while True:
            token = self._sample(logits[current])
            match = None
            for child in children.get(current, []):
                if nodes[child].token_id == token:
                    match = child
                    break
            if match is None:
                return accepted_rows, draft_tokens, token
            draft_tokens.append(token)
            accepted_rows.append(match)
            current = match
            if not children.get(current):
                return accepted_rows, draft_tokens, self._sample(logits[current])

    # -- main loop --------------------------------------------------------
    def generate(self) -> Iterator[int]:
        started = time.perf_counter()
        prompt = self.prompt_ids or [self.config.eos_token_id]

        # ---- prefill -----------------------------------------------------
        out = self.decoder.encode(
            prompt,
            self.state,
            positions=np.arange(len(prompt), dtype=np.float32),
            output_range=(0, len(prompt)),
            hidden_feature_layer_indices=self._spec_indices,
        )
        self.metrics.num_prefill_forward_passes += 1
        assert out.logits is not None
        root = self._sample(out.logits[-1])
        self.state.encode_accept(range(len(prompt)))
        self._last_features = self._capture_features(out.hidden_features, -1)
        self.metrics.first_token_seconds = time.perf_counter() - started
        self.generated.append(root)
        self.metrics.tokens_count_output = len(self.generated)
        yield root
        if root == self.config.eos_token_id:
            self.metrics.total_seconds = time.perf_counter() - started
            return

        # ---- decode ------------------------------------------------------
        while len(self.generated) < self.config.max_tokens:
            remaining = self.config.max_tokens - len(self.generated)
            use_spec = (
                self.speculator is not None
                and self._spec_state is not None
                and self.decoder.speculation_supported
                and remaining >= 2
            )

            if use_spec:
                shape = self.speculator.make_shape(remaining)
                if shape is None:
                    use_spec = False

            if use_spec:
                tree = self.speculator.propose_tree(
                    self._spec_state,
                    root,
                    self._last_features,
                    shape,
                    prng=self.prng,
                )
                nodes = tree.nodes
                topology = BatchTopology.from_nodes(nodes)
                topology.validate()
                positions = topology.positions(self.state.context_length)
                mask = topology.ancestor_mask(self.state.context_length)

                out = self.decoder.encode(
                    [n.token_id for n in nodes],
                    self.state,
                    positions=positions,
                    output_range=(0, len(nodes)),
                    hidden_feature_layer_indices=self._spec_indices,
                    mask=mask,
                )
                self.metrics.num_decode_forward_passes += 1
                assert out.logits is not None

                accepted_rows, draft_tokens, bonus = self._verify(nodes, out.logits)
                self.state.encode_accept(accepted_rows)
                self.speculator.encode_accept(self._spec_state, accepted_rows, out.hidden_features)
                self._last_features = self._capture_features(out.hidden_features, accepted_rows[-1])

                self.metrics.speculative_rounds += 1
                self.metrics.accepted_draft_tokens += len(draft_tokens)
                tokens = draft_tokens + [bonus]
            else:
                out = self.decoder.encode(
                    [root],
                    self.state,
                    positions=np.asarray([self.state.context_length], dtype=np.float32),
                    output_range=(0, 1),
                    hidden_feature_layer_indices=self._spec_indices,
                )
                self.metrics.num_decode_forward_passes += 1
                assert out.logits is not None
                self.state.encode_accept([0])
                bonus = self._sample(out.logits[0])
                self._last_features = self._capture_features(out.hidden_features, 0)
                tokens = [bonus]

            for token in tokens:
                if len(self.generated) >= self.config.max_tokens:
                    break
                self.generated.append(token)
                self.metrics.tokens_count_output = len(self.generated)
                yield token
                root = token
                if token == self.config.eos_token_id:
                    self.metrics.total_seconds = time.perf_counter() - started
                    return

        self.metrics.total_seconds = time.perf_counter() - started

    def run(self) -> list[int]:
        return list(self.generate())

    def __iter__(self) -> Iterator[int]:
        return self.generate()
