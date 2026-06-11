"""Sample-then-validate grammar sampling (phase-4 spec §4, approach B).

Ports llama.cpp's common/sampling.cpp strategy (grammar_first=false): sample
with the normal grammarless chain, check only the sampled token against the
grammar, and on rejection apply the grammar mask to the FULL vocabulary
before resampling. The full-vocab mask cannot empty the candidate set, so
the post-top-k abort (std::runtime_error "Unexpected empty grammar stack")
is structurally unreachable, and the common case costs one singleton
grammar check (~µs) instead of a ~152k-token mask (~30 ms) per token.

Chain parameters mirror create_completion's defaults in the pinned
llama-cpp-python 0.3.28 so that, absent a rescue, the seeded RNG stream —
and therefore the output — is identical to the grammarless path. A version
bump must re-check these constants. The chain's internal accept of a
subsequently-rescued token is harmless: with the default penalty parameters
the penalties sampler is a no-op (same simplification llama.cpp makes).
"""
from __future__ import annotations

import ctypes

import numpy as np

import llama_cpp
import llama_cpp._internals as internals

# create_completion defaults in llama-cpp-python 0.3.28.
CHAIN_DEFAULTS = {
    "repeat_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "top_k": 40,
    "typical_p": 1.0,
    "top_p": 0.95,
    "min_p": 0.05,
    "temp": 0.80,
}

_TOKEN_DATA_DTYPE = np.dtype(
    [("id", np.int32), ("logit", np.float32), ("p", np.float32)]
)


class SampleThenValidate:
    """Per-generation grammar-constrained sampler (one per generate call)."""

    def __init__(self, llm, grammar, seed: int | None) -> None:
        self._llm = llm
        self.rescued = False
        min_keep = 1  # upstream: max(1, n_probs) with n_probs = 0
        self._chain = internals.LlamaSampler()
        self._chain.add_penalties(
            penalty_last_n=llm.last_n_tokens_size,
            penalty_repeat=CHAIN_DEFAULTS["repeat_penalty"],
            penalty_freq=CHAIN_DEFAULTS["frequency_penalty"],
            penalty_present=CHAIN_DEFAULTS["presence_penalty"],
        )
        self._chain.add_top_k(CHAIN_DEFAULTS["top_k"])
        self._chain.add_typical(CHAIN_DEFAULTS["typical_p"], min_keep)
        self._chain.add_top_p(CHAIN_DEFAULTS["top_p"], min_keep)
        self._chain.add_min_p(CHAIN_DEFAULTS["min_p"], min_keep)
        self._chain.add_temp(CHAIN_DEFAULTS["temp"])
        self._chain.add_dist(
            llama_cpp.LLAMA_DEFAULT_SEED if seed is None else seed
        )
        self._grammar = internals.LlamaSampler()
        self._grammar.add_grammar(llm._model, grammar)

    def sample_token(self) -> int:
        """One grammar-valid token (or an EOG the grammar allows)."""
        tok = self._chain.sample(self._llm._ctx, -1)
        if self._grammar_allows(tok):
            if not self.is_eog(tok):
                self._grammar.accept(tok)
            return tok
        self.rescued = True
        tok = self._rescue()
        if not self.is_eog(tok):
            self._grammar.accept(tok)
        return tok

    def is_eog(self, tok: int) -> bool:
        return llama_cpp.llama_vocab_is_eog(self._llm._model.vocab, tok)

    def _grammar_allows(self, tok: int) -> bool:
        # Grammar apply masks invalid candidates to -inf; it validates EOG
        # against stack-emptiness, and never mutates parse state.
        data = (llama_cpp.llama_token_data * 1)(
            llama_cpp.llama_token_data(tok, 0.0, 0.0)
        )
        arr = llama_cpp.llama_token_data_array(data, 1, -1, False)
        llama_cpp.llama_sampler_apply(self._grammar.sampler, ctypes.byref(arr))
        return data[0].logit != float("-inf")

    def _rescue(self) -> int:
        """Grammar-first resample over the FULL vocab (never empty)."""
        n_vocab = self._llm._n_vocab
        logits = np.ctypeslib.as_array(
            self._llm._ctx.get_logits_ith(-1), shape=(n_vocab,)
        )
        buf = (llama_cpp.llama_token_data * n_vocab)()
        view = np.frombuffer(buf, dtype=_TOKEN_DATA_DTYPE)
        view["id"] = np.arange(n_vocab, dtype=np.int32)
        view["logit"] = logits
        view["p"] = 0.0
        arr = llama_cpp.llama_token_data_array(buf, n_vocab, -1, False)
        llama_cpp.llama_sampler_apply(self._grammar.sampler, ctypes.byref(arr))
        llama_cpp.llama_sampler_apply(self._chain.sampler, ctypes.byref(arr))
        return arr.data[arr.selected].id
