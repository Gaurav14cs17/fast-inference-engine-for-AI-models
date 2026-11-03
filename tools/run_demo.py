#!/usr/bin/env python3
"""Run the Python study port.

    cd python_study
    python tools/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when this file is run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import study_book  # noqa: E402,F401  (loads every numbered module in DAG order)

from study_book.chapter_02_configuration.config import hybrid_config, tiny_config  # noqa: E402
from study_book.chapter_02_configuration.quantization import RHT, TrellisCodec, quantize_group  # noqa: E402
from study_book.chapter_05_runtime.engine import ChatConfig, ChatReplyConfig, Engine, EngineConfig  # noqa: E402
from study_book.chapter_05_runtime.tokenizer import ChatMessage  # noqa: E402


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_chat(config, label: str, seed: int = 7) -> None:
    banner(label)
    engine = Engine.new(EngineConfig())
    info = engine.load_random(config, seed=seed)
    session = engine.chat(info, ChatConfig())
    messages = [
        ChatMessage.system("You are a helpful assistant."),
        ChatMessage.user("Tell me a short story about a robot."),
    ]
    replies = session.reply(messages, ChatReplyConfig())
    reply = replies[0]
    print("text:", repr(reply.message.content))
    stats = reply.stats
    print(
        "metrics:",
        {
            "prefill": stats.num_prefill_forward_passes,
            "decode": stats.num_decode_forward_passes,
            "tokens": stats.tokens_count_output,
            "tokens/forward": round(stats.tokens_per_forward_pass, 3),
            "spec_rounds": stats.speculative_rounds,
            "accepted_draft": stats.accepted_draft_tokens,
            "seconds": round(stats.total_seconds, 4),
        },
    )

    # Streaming API
    stream = session.reply_with_stream(messages, ChatReplyConfig())
    pieces = []
    final = None
    for chunk in stream:
        if chunk.finished:
            final = chunk
        else:
            pieces.append(chunk.text)
    print("stream text:", repr("".join(pieces)))
    if final and final.stats:
        print("stream metrics:", final.stats.tokens_per_forward_pass, "tokens/forward")


def quantization_demo() -> None:
    banner("quantization primitives (file 06)")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 64)).astype(np.float32)

    q = quantize_group(x, group_size=32, bits=4)
    err = np.abs(q.dequantize() - x).mean()
    print(f"int4 group-32: {q.bits_per_weight:.2f} bits/weight, mean abs error {err:.4f}")

    rht = RHT(64, seed=1)
    rotated = rht.apply(x)
    print("RHT is norm-preserving:", np.allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(rotated, axis=-1), atol=1e-4))

    codec = TrellisCodec.encode(x[0])
    decoded = TrellisCodec.decode(codec)
    print("trellis toy codec: 2 bits/step, mean abs error", round(float(np.abs(decoded - x[0]).mean()), 4))


def verification_demo() -> None:
    """Show the tree-accept mechanic without needing trained draft weights."""
    banner("tree verification with synthetic target logits (file 28)")
    from study_book.chapter_04_generation.weaver import ProposalNode  # noqa: E402
    from study_book.chapter_05_runtime.stream import GenerationStream  # noqa: E402

    # A chain: root(t=10) -> 11 -> 12 -> 13, plus a bonus token 14.
    nodes = [ProposalNode(10, 0, 0.0, -1, [1])]
    for depth, token in enumerate([11, 12, 13], start=1):
        nodes.append(ProposalNode(token, depth, 0.0, depth - 1, [depth + 1] if depth < 3 else []))
    nodes[-1].children = []

    engine = Engine.new(EngineConfig())
    info = engine.load_random(tiny_config(), seed=3)
    model = engine.language_model(info)
    stream = GenerationStream(model, prompt_ids=[])

    vocab = model.config.decoder_config.vocab_size
    logits = np.full((len(nodes), vocab), -1e9, dtype=np.float32)
    # row 0 predicts child 11, row 1 predicts 12, row 2 predicts 13, row 3 predicts bonus 14
    logits[0, 11] = 0.0
    logits[1, 12] = 0.0
    logits[2, 13] = 0.0
    logits[3, 14] = 0.0

    accepted_rows, draft_tokens, bonus = stream._verify(nodes, logits)
    print("accepted rows:", accepted_rows)
    print("accepted draft tokens:", draft_tokens)
    print("bonus token:", bonus)
    print("cache advances by:", len(accepted_rows), "rows, producing", len(draft_tokens) + 1, "tokens")

    print(
        "\nNote: with random weights the independent DFlash draft rarely matches the\n"
        "target, so the live chat above accepts 0 draft tokens. Real DFlash weights\n"
        "are trained to match the target; the verification path is the same."
    )



def main() -> None:
    print("NOTE: this demo uses random weights, so the generated text is gibberish.")
    print("The point is the dataflow: embeddings -> layers -> cache -> sampling -> tree verify.")
    run_chat(tiny_config(), "attention-only model + DFlash/Weaver speculation")
    run_chat(hybrid_config(), "hybrid model (attention + DeltaNet), no speculation")
    quantization_demo()
    verification_demo()
    print("\nDone. The numbered modules are the DAG order; see docs/00_DAG.md.")


if __name__ == "__main__":
    main()
