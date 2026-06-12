# Findings Restructure — Living Index + Verbatim Phase Files Design

**Date:** 2026-06-12
**Status:** Approved (pending final spec review)
**Builds on:** Phase 5 (`docs/superpowers/specs/2026-06-11-lce-phase5-savings-tuning-design.md`, merged at `d0fe6f3`).
**Scope:** Documentation only — no code, no tests, no behavior changes. Split the 402-line `docs/findings.md` into a short living index plus per-phase history files moved verbatim, so per-phase maintenance cost stays constant as phases accumulate.

## 1. Problem

`docs/findings.md` serves two readers at once: future-you/agents starting the
next phase (need "what's currently true", fast) and external readers (the
chronological narrative is the value). Both jobs live interleaved in one
growing file:

- §1–7 are *topical* slices of phases 1–2; §8–10 switched to
  one-big-section-per-phase — structurally inconsistent.
- Each phase appends ~100–120 lines **and** back-patches
  "Actualización fase X: ver §Y" blockquotes into old sections — the
  growth pattern that does not scale.
- Superseded numbers (phase-4 absolutes that §10 declares no longer
  comparable) sit next to current ones with no distinction.

Constraint (project ethos): historical text is **append-only** — it may move
to other files verbatim, but is never rewritten, condensed, or translated.

## 2. Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| Structure | `docs/findings.md` becomes a ~70-line living index; history moves to `docs/findings/fase-*.md` | Only structure where per-phase maintenance cost is constant; serves both readers (index = current truth, phase files = untouched narrative) |
| Index location | `docs/findings.md` keeps the path | Specs, plans, and habit all point at it; "findings" still accurately names an index-of-findings; a rename adds a hop for nothing |
| Move fidelity | Phase-file bodies are byte-identical to the moved lines; headings keep their numbers (`## 9. Fase 4 — …`, `### 10.4 …`) | Append-only guarantee; section numbers become permanent IDs — every historical citation ("findings.md §10", "ver §9.4") resolves via the index map |
| Phase grouping | `fase-1-2.md` (§1–7), `fase-3.md` (§8), `fase-4.md` (§9), `fase-5.md` (§10) | §1–7 are one era (phases 1–2) and only make sense together; §8–10 already are phase chunks |
| Per-file header | 2–3 added lines above the moved content: phase, date, merge commit | Additions, not edits — append-only holds |
| Supersession | Recorded in the index only; the back-patching pattern is retired | The two §1/§8 blockquotes added in phase 5 move verbatim (they are history now); no new ones are ever added |
| Language | Index in Spanish, matching findings.md | One document, one language; specs/plans stay English per existing convention |
| Tooling | None (no link checker, no generator) | YAGNI for 5 files |

## 3. Changes by File

| File | Change |
|---|---|
| `docs/findings/fase-1-2.md` | New: header + lines 10–126 of current findings.md (§1–§7), verbatim |
| `docs/findings/fase-3.md` | New: header + lines 127–186 (§8), verbatim |
| `docs/findings/fase-4.md` | New: header + lines 187–288 (§9), verbatim |
| `docs/findings/fase-5.md` | New: header + lines 289–402 (§10), verbatim |
| `docs/findings.md` | Replaced with the living index (§4 below) |

No other files change. Historical specs/plans that cite "findings.md §N" are
themselves append-only and stay untouched; their citations resolve through
the index's § map.

## 4. The Living Index (new findings.md, ~70 lines, Spanish)

1. **Header** — title, última actualización, estado del proyecto, entorno
   (the current machine/model block — living info: it already changed once,
   39→41 corpus files).
2. **Estado actual** — only numbers true *now*, each tagged with source run
   and phase file: ahorro lean 32.7% intra-run (`p5-baseline`, fase 5);
   76.6% cross-config con costo de calidad rechazado; TTFT lean_grammar
   ~291 ms; hit-rate 64/90; compresión corpus 94.0% (41 archivos); defaults
   `k=3`/sin cap; system prompt 48 tokens. Standing caveats: fase-4
   absolutes no longer comparable (system prompt + corpus changed); corpus
   contaminated by experiment docs (39→41).
3. **Preguntas abiertas** — carried from §10.5/§10.6: per-document skeleton
   compression (the structural ceiling), k=1 retrieval near-tie
   instability, corpus contamination.
4. **Índice de fases** — table: fase | secciones | one-line summary | link.
   This is the §→file map; every § number 1–10 appears exactly once.
5. **Convención** — phase files are append-only history; phase N adds
   `docs/findings/fase-N.md` continuing the § numbering and edits only this
   index; supersession is recorded here, never back-patched. Note the split
   date and that `git log --follow docs/findings/fase-*.md` recovers
   pre-split history.

All numbers in §2 of the index come from the committed record
(`docs/superpowers/plans/p5-sweep-results.txt`, findings §10) — no new
measurements.

## 5. Phase-File Header Format

```markdown
> Fase N (YYYY-MM-DD) — movido íntegro desde `docs/findings.md` §X–§Y
> el 2026-06-12; historia previa: `git log --follow`. Rama: <branch>,
> merge: <commit>.
```

Then a blank line, then the moved content starting at its original `## N.`
heading. Nothing inside the moved content changes — verifiable.

## 6. Verification

1. Byte-identity: for each phase file, stripping the added header lines and
   diffing against the corresponding line range of
   `git show HEAD~1:docs/findings.md` yields an empty diff.
2. Index completeness: § numbers 1–10 each appear exactly once in the index
   map; the five files in §3 exist; no content lines of the old file are
   orphaned (sum of moved ranges = lines 10–402).
3. `uv run pytest` untouched-green (docs-only change; run as a smoke check).
4. One commit, `docs:` prefix.

## 7. Going Forward (the maintainability contract)

A phase's final documentation task changes from "append §N + back-patch
pointers" to:

1. Create `docs/findings/fase-N.md` with the next § number.
2. Update the index: estado actual, preguntas abiertas, fase table row.

Cost is constant in the number of phases. The index is the only file that
is ever edited; phase files are write-once.

## 8. Out of Scope

Rewriting, condensing, or translating any moved text; editing historical
specs/plans; renaming the document; link-checking or generation tooling;
any code or test changes.
