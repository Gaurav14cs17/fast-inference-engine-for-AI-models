"""07 — safetensors loading and parameter trees.

Mirrors ``crates/uzu-engine/src/safetensors.rs``.

The Rust ``ParameterTree`` gives typed access to tensors and tracks which ones
were read (``assert_all_tensors_read``).  The Python version also supports a
random-init fallback so the demo can build a model without a weight file.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from study_book.chapter_01_foundations.dtypes import DataType


def load_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    try:
        from safetensors.numpy import load_file  # type: ignore

        return {k: np.asarray(v) for k, v in load_file(str(path)).items()}
    except Exception:
        return _load_safetensors_manual(path)


def _load_safetensors_manual(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    data = raw[8 + header_len :]
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.float32,
            "I8": np.int8,
            "U8": np.uint8,
            "I32": np.int32,
            "U32": np.uint32,
        }[info["dtype"]]
        start, end = info["data_offsets"]
        arr = np.frombuffer(data[start:end], dtype=dtype).reshape(info["shape"]).copy()
        out[name] = arr
    return out


def save_safetensors(path: str | Path, tensors: dict[str, np.ndarray]) -> None:
    from safetensors.numpy import save_file  # type: ignore

    save_file({k: np.asarray(v) for k, v in tensors.items()}, str(path))


@dataclass
class ParameterTree:
    """Nested view over a flat ``{name: ndarray}`` mapping."""

    flat: dict[str, np.ndarray]
    prefix: str = ""
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    allow_random: bool = True
    read_keys: set[str] = field(default_factory=set)

    def _key(self, name: str) -> str:
        return f"{self.prefix}.{name}" if self.prefix else name

    def subtree(self, path: str) -> "ParameterTree":
        return ParameterTree(
            self.flat,
            self._key(path),
            self.rng,
            self.allow_random,
            self.read_keys,
        )

    def has(self, name: str) -> bool:
        key = self._key(name)
        return key in self.flat or any(k.startswith(key + ".") for k in self.flat)

    def read(
        self,
        name: str,
        shape: Optional[tuple[int, ...] | list[int]] = None,
        dtype: DataType = DataType.F32,
        init: str = "normal",
    ) -> np.ndarray:
        key = self._key(name)
        if key not in self.flat:
            if not self.allow_random:
                raise KeyError(f"missing tensor {key}")
            if shape is None:
                raise KeyError(f"missing tensor {key} and no shape supplied for random init")
            shape = tuple(int(x) for x in shape)
            if init == "ones":
                value = np.ones(shape, dtype=np.float32)
            elif init == "zeros":
                value = np.zeros(shape, dtype=np.float32)
            else:
                value = (self.rng.standard_normal(shape) * 0.02).astype(np.float32)
            self.flat[key] = value
        arr = self.flat[key]
        if shape is not None and tuple(arr.shape) != tuple(shape):
            raise ValueError(f"{key}: expected shape {tuple(shape)}, got {arr.shape}")
        if dtype == DataType.INT8:
            arr = arr.astype(np.int8)
        elif dtype.is_float:
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(dtype.numpy_dtype)
        self.flat[key] = arr
        self.read_keys.add(key)
        return arr

    def read_optional(
        self,
        name: str,
        shape: Optional[tuple[int, ...] | list[int]] = None,
        dtype: DataType = DataType.F32,
        init: str = "normal",
    ) -> Optional[np.ndarray]:
        if not self.has(name):
            return None
        return self.read(name, shape, dtype, init)

    def names(self) -> list[str]:
        return sorted(self.flat)

    def assert_all_tensors_read(self) -> None:
        unread = set(self.flat) - self.read_keys
        if unread:
            print(f"[parameter tree] warning: {len(unread)} tensors were never read:")
            for name in sorted(unread)[:10]:
                print("  -", name)
