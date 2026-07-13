"""
ONNX-based embedding for all-MiniLM-L6-v2.

Model interface:
  Inputs:  input_ids [batch, 128] int64
           attention_mask [batch, 128] int64
           token_type_ids [batch, 128] int64  ← optional, not all exports include it
  Output:  last_hidden_state [batch, 128, 384] float32

Pipeline:
  1. Tokenize with padding/truncation to 128 tokens
  2. Run ONNX inference on CPU
  3. Mean-pool only non-padding token positions
  4. L2-normalize each embedding

Input names and output index are inspected dynamically at load time so the
class works with both 2-input and 3-input ONNX exports without modification.

Runs on CPU to keep VRAM free for the local LLM.
"""
from __future__ import annotations

from typing import List, Dict, Set

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class ONNXEmbedder:
    """
    Embeds text strings using all-MiniLM-L6-v2 via ONNX Runtime.

    Both embed() and batch_embed() return L2-normalised float32 arrays.
    Since vectors are normalised, cosine similarity == dot product.
    """

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        """
        Load ONNX model and HuggingFace tokenizer.

        Inspects actual input/output names at load time — does not assume
        token_type_ids is present or that last_hidden_state is output[0].

        Args:
            model_path:      Path to model.onnx
            tokenizer_path:  Path to tokenizer.json
        """
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding(length=128)
        self._tokenizer.enable_truncation(max_length=128)

        # Fix #1: Inspect actual input names — don't assume token_type_ids exists
        self._input_names: Set[str] = {inp.name for inp in self._session.get_inputs()}

        # Fix #2: Inspect actual output names — don't assume last_hidden_state is outputs[0]
        output_names = [out.name for out in self._session.get_outputs()]
        if "last_hidden_state" in output_names:
            self._hidden_state_idx = output_names.index("last_hidden_state")
        else:
            # Fallback: use first output and warn
            self._hidden_state_idx = 0
            import warnings
            warnings.warn(
                f"'last_hidden_state' not found in model outputs {output_names}. "
                "Falling back to outputs[0]. Verify this is correct for your export.",
                RuntimeWarning,
                stacklevel=2,
            )

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single string.
        Returns L2-normalised float32 array of shape (384,).
        """
        return self.batch_embed([text])[0]

    def batch_embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings.
        Returns L2-normalised float32 array of shape (N, 384).
        """
        encodings = self._tokenizer.encode_batch(texts)

        input_ids      = np.array([e.ids           for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        # Build feed dict from actual model inputs only
        feed: Dict[str, np.ndarray] = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self._session.run(None, feed)

        # Select last_hidden_state by verified index, not positional assumption
        last_hidden = outputs[self._hidden_state_idx]  # (batch, seq_len, 384)

        # Mean pool: only non-padding positions contribute
        mask    = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed  = (last_hidden * mask).sum(axis=1)
        counts  = mask.sum(axis=1).clip(min=1e-9)
        pooled  = summed / counts

        # L2 normalise each row
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (pooled / norms).astype(np.float32)