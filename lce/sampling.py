"""Post-top-k grammar sampler chain (phase-4 spec §4).

Mirrors `llama_cpp.llama.Llama._init_sampler` of the pinned llama-cpp-python
0.3.28 (llama.py:735-779) with exactly one change: the grammar sampler is
added AFTER min_p — masking the <=top_k surviving candidates — instead of
before top_k, where it masks the full ~152k-token vocab at ~+30 ms/token
(measured; see docs/findings.md §9). Identical seeded output was measured
for both orders on the routing task.

Any llama-cpp-python version bump must re-diff this override against the
upstream method. Only the default sampling branch is mirrored (temp > 0, no
mirostat, no logits_processor) — the engine never uses the others, and the
override refuses them rather than silently mis-ordering.
"""
from __future__ import annotations

import llama_cpp._internals as internals
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama import LogitsProcessorList


class PostTopKGrammarLlama(Llama):
    # Set per call by Engine.generate before create_completion; True restores
    # the upstream grammar-first chain (the structurally safe slow path).
    grammar_first: bool = False

    def _init_sampler(
        self,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        temp: float = 0.80,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_eta: float = 0.1,
        mirostat_tau: float = 5.0,
        penalize_nl: bool = True,
        logits_processor: LogitsProcessorList | None = None,
        grammar: LlamaGrammar | None = None,
    ):
        if grammar is None or self.grammar_first:
            return super()._init_sampler(
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                temp=temp,
                repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_eta=mirostat_eta,
                mirostat_tau=mirostat_tau,
                penalize_nl=penalize_nl,
                logits_processor=logits_processor,
                grammar=grammar,
            )
        if temp <= 0.0 or mirostat_mode != 0 or logits_processor is not None:
            raise NotImplementedError(
                "PostTopKGrammarLlama mirrors only the default sampling branch"
                " (temp > 0, no mirostat, no logits_processor); set"
                " grammar_first=True for other configurations."
            )
        sampler = internals.LlamaSampler()
        sampler.add_penalties(
            penalty_last_n=self.last_n_tokens_size,
            penalty_repeat=repeat_penalty,
            penalty_freq=frequency_penalty,
            penalty_present=presence_penalty,
        )
        min_keep = 1  # upstream: max(1, n_probs) with n_probs = 0
        sampler.add_top_k(top_k)
        sampler.add_typical(typical_p, min_keep)
        sampler.add_top_p(top_p, min_keep)
        sampler.add_min_p(min_p, min_keep)
        sampler.add_grammar(self._model, grammar)  # moved: post-truncation mask
        sampler.add_temp(temp)
        sampler.add_dist(self._seed)
        return sampler
