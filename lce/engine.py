"""llama-cpp-python wrapper with optional GBNF-constrained decoding.

Foundation spec §4 (engine) and §8 (error handling). Token counts come from
llama.cpp's own tokenizer; TTFT is the timestamp of the first streamed token.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_ms: float


class EngineLoadError(RuntimeError):
    """The GGUF model could not be loaded."""


class Engine:
    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise EngineLoadError(
                f"Model file not found: {model_path}. "
                "Run scripts/setup_env.sh to download it."
            )
        from llama_cpp import Llama  # deferred: slow import, needs native lib

        try:
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            raise EngineLoadError(
                f"Failed to load {model_path.name} "
                f"(n_gpu_layers={n_gpu_layers}). If VRAM is exhausted, retry "
                f"with fewer layers, e.g. Engine(..., n_gpu_layers=20). "
                f"Original error: {exc}"
            ) from exc

        self._grammar_cache: dict[Path, "LlamaGrammar"] = {}

    def _load_grammar(self, grammar_path: str | Path) -> "LlamaGrammar":
        from llama_cpp import LlamaGrammar

        path = Path(grammar_path)
        if path not in self._grammar_cache:
            self._grammar_cache[path] = LlamaGrammar.from_string(path.read_text())
        return self._grammar_cache[path]

    def generate(
        self,
        prompt: str,
        *,
        grammar_path: str | Path | None = None,
        max_tokens: int = 256,
    ) -> GenerationResult:
        """Stream a completion and measure it.

        Measurement semantics (foundation spec §5):
        - prompt_tokens: llama.cpp tokenization (special=True) of `prompt`,
          identical to what create_completion evaluates.
        - completion_tokens: count of streamed content chunks (one per token);
          the trailing finish_reason sentinel chunk is excluded.
        - ttft_ms: time from generation start to the first content chunk.
          Grammar compilation is cached per path and excluded by design.
        - total_ms: time from generation start to stream end. Falls back as
          ttft_ms when zero tokens are generated (immediate EOS).

        The llama context is reset before each call: Llama.generate otherwise
        reuses the KV state for common prompt prefixes, which made TTFT depend
        on call order (identical prompts measured ~10x faster on the second
        call). Resetting makes every transaction pay its full prompt eval, so
        ttft_ms is comparable across arms and reps.
        """
        grammar = self._load_grammar(grammar_path) if grammar_path is not None else None
        # special=True matches _create_completion's internal tokenization of
        # string prompts, so this count equals what the model actually evaluates.
        prompt_tokens = len(self._llm.tokenize(prompt.encode("utf-8"), special=True))
        self._llm.reset()  # defeat prefix-match KV reuse (see docstring)
        pieces: list[str] = []
        completion_tokens = 0
        ttft_ms: float | None = None
        start = time.perf_counter()
        for chunk in self._llm.create_completion(
            prompt, max_tokens=max_tokens, grammar=grammar, stream=True
        ):
            choice = chunk["choices"][0]
            if choice["finish_reason"] is not None:
                continue  # trailing sentinel chunk, not a generated token
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            pieces.append(choice["text"])
            completion_tokens += 1
        total_ms = (time.perf_counter() - start) * 1000.0
        return GenerationResult(
            text="".join(pieces),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=total_ms if ttft_ms is None else ttft_ms,
            total_ms=total_ms,
        )


_VALID_ACTIONS = {"open_file", "search_code", "explain", "none"}


def validate_routing_output(text: str) -> bool:
    """True iff `text` is exactly the routing JSON enforced by router.gbnf.

    A malformed response is data (format_success=0), never an exception.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(obj, dict)
        and set(obj) == {"action", "target", "confidence"}
        and obj["action"] in _VALID_ACTIONS
        and isinstance(obj["target"], str)
        and isinstance(obj["confidence"], (int, float))
        and not isinstance(obj["confidence"], bool)
        and 0.0 <= obj["confidence"] <= 1.0
    )
