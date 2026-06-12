# Findings Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 402-line `docs/findings.md` into a ~70-line living index plus four per-phase history files moved **verbatim**, so per-phase maintenance cost stays constant.

**Architecture:** Docs-only. The four phase files are byte-identical extractions of line ranges from the current findings.md (§1–7 → `fase-1-2.md`, §8 → `fase-3.md`, §9 → `fase-4.md`, §10 → `fase-5.md`), each prefixed with a 2-line provenance blockquote + blank line. findings.md is replaced by the living index (Spanish): estado actual, preguntas abiertas, §→file map, convención. Byte-identity is mechanically verified before the commit.

**Tech Stack:** bash (`sed`, `diff`), git. No Python changes.

**Spec:** `docs/superpowers/specs/2026-06-12-findings-restructure-design.md`

**Single-commit rule (spec §6.4):** Unlike normal plans, do NOT commit per task. All changes land in ONE `docs:` commit in Task 4, after verification. The spec's post-commit check (`git show HEAD~1:docs/findings.md`) only works if the split is exactly one commit.

**Fixed facts (verified 2026-06-12, findings.md at `d0fe6f3`, 402 lines):**

| § range | lines | phase file | publishing commit(s) |
|---|---|---|---|
| §1–§7 | 10–126 | `fase-1-2.md` | `9ab5713` (2026-06-10) |
| §8 | 127–186 | `fase-3.md` | `bda3bff` (2026-06-10) |
| §9 | 187–288 | `fase-4.md` | `9298540` (2026-06-11) |
| §10 | 289–402 | `fase-5.md` | `1d25b1b`, ajuste `d0fe6f3` |

Lines 1–8 (title/header block) and line 9 (`---`) are absorbed by the new index; they are not "content" and move nowhere.

---

### Task 1: Snapshot + create the four phase files (verbatim extraction)

**Files:**
- Create: `docs/findings/fase-1-2.md`
- Create: `docs/findings/fase-3.md`
- Create: `docs/findings/fase-4.md`
- Create: `docs/findings/fase-5.md`

- [ ] **Step 1: Verify the precondition and snapshot the pre-split file**

```bash
cd /home/irvint/experiment
git diff --quiet -- docs/findings.md && wc -l docs/findings.md
cp docs/findings.md /tmp/findings-pre-split.md
mkdir -p docs/findings
```

Expected: no diff output (clean working tree for this file) and `402 docs/findings.md`. **If the line count is not 402, STOP** — the line ranges below are wrong; re-derive them with `grep -n "^## " docs/findings.md` before continuing.

- [ ] **Step 2: Write the four provenance headers**

Each file gets EXACTLY 2 blockquote lines + 1 blank line (content starts at line 4 — the verification in Task 3 depends on this). Create the files with only their headers:

```bash
cd /home/irvint/experiment

cat > docs/findings/fase-1-2.md <<'EOF'
> Fases 1–2 (2026-06-10) — §1–§7, movido íntegro desde `docs/findings.md` el 2026-06-12.
> Publicado en `9ab5713`; historia previa: `git log -- docs/findings.md`.

EOF

cat > docs/findings/fase-3.md <<'EOF'
> Fase 3 (2026-06-10) — §8, movido íntegro desde `docs/findings.md` el 2026-06-12.
> Publicado en `bda3bff`; historia previa: `git log -- docs/findings.md`.

EOF

cat > docs/findings/fase-4.md <<'EOF'
> Fase 4 (2026-06-11) — §9, movido íntegro desde `docs/findings.md` el 2026-06-12.
> Publicado en `9298540`; historia previa: `git log -- docs/findings.md`.

EOF

cat > docs/findings/fase-5.md <<'EOF'
> Fase 5 (2026-06-11) — §10, movido íntegro desde `docs/findings.md` el 2026-06-12.
> Publicado en `1d25b1b` (ajuste `d0fe6f3`); historia previa: `git log -- docs/findings.md`.

EOF
```

(Note: the spec §5 sketch mentions `git log --follow`; this plan deliberately uses `git log -- docs/findings.md` instead — `--follow` does not reliably trace a 1→4 file split, while the old path's log always works.)

- [ ] **Step 3: Append the verbatim content from the snapshot**

```bash
cd /home/irvint/experiment
sed -n '10,126p'  /tmp/findings-pre-split.md >> docs/findings/fase-1-2.md
sed -n '127,186p' /tmp/findings-pre-split.md >> docs/findings/fase-3.md
sed -n '187,288p' /tmp/findings-pre-split.md >> docs/findings/fase-4.md
sed -n '289,402p' /tmp/findings-pre-split.md >> docs/findings/fase-5.md
```

- [ ] **Step 4: Spot-check the seams**

```bash
cd /home/irvint/experiment
head -4 docs/findings/fase-1-2.md | tail -1   # "## 1. Resultados del primer benchmark honesto"
head -4 docs/findings/fase-3.md  | tail -1    # "## 8. Fase 3 — Compresión markdown outline-only..."
head -4 docs/findings/fase-4.md  | tail -1    # "## 9. Fase 4 — Causa raíz de la latencia..."
head -4 docs/findings/fase-5.md  | tail -1    # "## 10. Fase 5 — Ajuste de ahorro de prompt..."
tail -1 docs/findings/fase-5.md               # "después `--k`/`--doc-cap` según la tabla (sin `--reindex`)."
```

Expected: each file's line 4 is its section's original `## N.` heading; fase-5 ends with the reproduction line. **Do not commit yet.**

---

### Task 2: Replace findings.md with the living index

**Files:**
- Modify: `docs/findings.md` (full replacement)

- [ ] **Step 1: Write the new findings.md**

Replace the ENTIRE content of `docs/findings.md` with exactly:

```markdown
# LCE — Hallazgos (Findings)

**Última actualización:** 2026-06-12
**Estado del proyecto:** Fases 1–5 fusionadas en `main`.
**Entorno:** EndeavourOS, i5-13420H, 16 GiB RAM, RTX 3050 6 GiB VRAM, CUDA 13.3,
llama-cpp-python 0.3.28 (cuBLAS), Qwen2.5-3B-Instruct Q4_K_M, ChromaDB 1.5.9 (embeddings CPU).
**Corpus:** 41 archivos (este repositorio; incluye spec y plan de fase 5 — ver preguntas abiertas).

Este archivo es el **índice vivo**: el estado actual y el mapa de la historia.
La historia completa vive en `docs/findings/fase-*.md`, movida íntegra desde
este archivo el 2026-06-12 (historia previa: `git log -- docs/findings.md`).

## Estado actual (lo que es cierto hoy)

Fuente: barrido de fase 5 — seed 42, CA, 270 transacciones/config
(`docs/superpowers/plans/p5-sweep-results.txt`; análisis en
[fase-5.md](findings/fase-5.md) §10).

| métrica | valor | fuente |
|---|---|---|
| Ahorro de prompt lean (intra-run, a paridad de calidad) | **32.7%** | `p5-baseline` (k=3, sin cap) |
| Ahorro cross-config (lean k=1 vs naive k=3) | 76.6% — **rechazado**: cuesta 6/90 hits | §10.5 |
| Target hit-rate lean_grammar | 64/90 (0.71) | `p5-baseline` |
| TTFT lean_grammar | 291.3 ms (p50 276.9) | `p5-baseline` |
| Compresión del corpus (chars) | 94.0% (41 archivos) | `p5-baseline` |
| System prompt | 48 tokens (83 hasta fase 4) | fase 5 |
| Defaults de la librería y el CLI | `k=3`, sin `doc_cap` | §10.4: ningún candidato pasó la barra |
| Calidad de ruteo lean vs naive (k=3) | lean gana: 64 vs 55 hits (naive fmt 0.82) | §10.6 |

Advertencias vigentes:

- Los números absolutos de fases ≤4 **ya no son comparables**: el system
  prompt se recortó en fase 5 (todos los brazos) y el corpus pasó de 39 a 41
  archivos.
- **>80% de ahorro sin pérdida de calidad no se alcanzó**; el techo es
  estructural — andamiaje fijo ~63 tokens + esqueleto completo ~179
  tokens/doc (§10.5).

## Preguntas abiertas

- Comprimir el esqueleto **por documento** (el techo estructural de §10.5):
  ¿outline más agresivo para código, u otra composición del prompt?
- Inestabilidad de retrieval con k=1: near-ties (Δdistancia 0.005) hacen
  flip del doc top-1 entre procesos (§10.6).
- Contaminación del corpus: los docs del propio experimento (spec/plan de
  fase 5) compiten en retrieval con el código que describen (§10.6).

## Índice de fases

Los números de sección (§) son IDs permanentes: una cita histórica como
«findings.md §9.4» se resuelve con esta tabla.

| fase | secciones | resumen | archivo |
|---|---|---|---|
| 1–2 | §1–§7 | Primer benchmark honesto: ahorro 29.4%, el caché de prefijos sesgaba el TTFT ~10×, la compresión depende del tipo de documento | [fase-1-2.md](findings/fase-1-2.md) |
| 3 | §8 | Markdown outline-only: corpus 93.4% (chars), ahorro 43.5%; batería de 30 consultas sembrada | [fase-3.md](findings/fase-3.md) |
| 4 | §9 | Causa raíz de la latencia con gramática (orden de samplers → muestrear-y-validar); gate de batería y estado de máquina | [fase-4.md](findings/fase-4.md) |
| 5 | §10 | Métrica target hit-rate; barrido k/doc_cap sin ganador (defaults quedan k=3/sin cap); veredicto >80%: no alcanzado | [fase-5.md](findings/fase-5.md) |

## Convención

- Los archivos de fase son **historia append-only**: se mueven íntegros y no
  se reescriben nunca.
- Fase N nueva ⇒ crear `docs/findings/fase-N.md` (continúa la numeración §)
  y editar **solo este índice** (estado actual, preguntas abiertas, tabla).
- La supersesión se registra aquí; nunca se retro-parchan archivos de fase.
```

- [ ] **Step 2: Sanity-render check**

```bash
cd /home/irvint/experiment
wc -l docs/findings.md
grep -c "findings/fase-" docs/findings.md
```

Expected: ~75 lines; 5 links (one in "Estado actual", four in the table). **Do not commit yet.**

---

### Task 3: Verify byte-identity and index completeness

**Files:** none modified — verification only.

- [ ] **Step 1: Byte-identity of the moved content**

Each phase file minus its 3 header lines must be byte-identical to its source range:

```bash
cd /home/irvint/experiment
ok=1
for spec in "fase-1-2:10:126" "fase-3:127:186" "fase-4:187:288" "fase-5:289:402"; do
  name=${spec%%:*}; rest=${spec#*:}; a=${rest%%:*}; b=${rest#*:}
  if diff <(sed -n "${a},${b}p" /tmp/findings-pre-split.md) \
          <(tail -n +4 "docs/findings/${name}.md") >/dev/null; then
    echo "${name}: BYTE-IDENTICAL"
  else
    echo "${name}: DIFFERS"; ok=0
  fi
done
[ "$ok" = 1 ] && echo "ALL VERBATIM" || echo "FAIL"
```

Expected: four `BYTE-IDENTICAL` lines and `ALL VERBATIM`. **If any file DIFFERS, fix the extraction (Task 1) — never edit the moved content to make it match.**

- [ ] **Step 2: No orphaned lines**

The moved ranges must cover lines 10–402 of the old file with no gaps:
10–126, 127–186, 187–288, 289–402 → contiguous, sum = 393 lines.

```bash
cd /home/irvint/experiment
total=$(( (126-10+1) + (186-127+1) + (288-187+1) + (402-289+1) ))
echo "moved=$total expected=393"
```

Expected: `moved=393 expected=393`.

- [ ] **Step 3: § map completeness**

Every § number 1–10 must appear exactly once in the index table's "secciones" column (§1–§7 covers 1..7):

```bash
cd /home/irvint/experiment
grep -E "^\| (1–2|3|4|5) \| §" docs/findings.md
```

Expected: exactly 4 rows: `§1–§7`, `§8`, `§9`, `§10` — together covering 1–10 with no repeats.

- [ ] **Step 4: Unit-suite smoke check (docs-only change)**

```bash
cd /home/irvint/experiment
uv run pytest -q
```

Expected: `115 passed` (4 GPU deselected). Nothing in `lce/` or `tests/` reads findings.md, so any failure means an environment problem, not this change.

---

### Task 4: The single commit

**Files:** stages everything from Tasks 1–2.

- [ ] **Step 1: Review the staged shape**

```bash
cd /home/irvint/experiment
git add docs/findings.md docs/findings/
git status --short
```

Expected: exactly `M  docs/findings.md` plus `A  docs/findings/fase-1-2.md`, `A  docs/findings/fase-3.md`, `A  docs/findings/fase-4.md`, `A  docs/findings/fase-5.md`. Nothing else.

- [ ] **Step 2: Commit (the only commit of this plan)**

```bash
cd /home/irvint/experiment
git commit -m "docs: split findings.md into living index + verbatim phase files

§1–7 → docs/findings/fase-1-2.md, §8 → fase-3.md, §9 → fase-4.md,
§10 → fase-5.md — bodies byte-identical to the pre-split file (verified);
the only additions are 2-line provenance headers. findings.md becomes the
living index: estado actual (run-tagged numbers), preguntas abiertas,
§→file map, convención. Back-patching of 'Actualización fase X' pointers
is retired; supersession now lives in the index only.

Spec: docs/superpowers/specs/2026-06-12-findings-restructure-design.md

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Post-commit spec verification (spec §6.1)**

The spec's own acceptance check, now that HEAD~1 is the pre-split state:

```bash
cd /home/irvint/experiment
for spec in "fase-1-2:10:126" "fase-3:127:186" "fase-4:187:288" "fase-5:289:402"; do
  name=${spec%%:*}; rest=${spec#*:}; a=${rest%%:*}; b=${rest#*:}
  diff <(git show HEAD~1:docs/findings.md | sed -n "${a},${b}p") \
       <(tail -n +4 "docs/findings/${name}.md") >/dev/null \
    && echo "${name}: OK" || echo "${name}: FAIL"
done
```

Expected: four `OK` lines. If any FAIL, the commit is wrong — fix forward with a follow-up commit is NOT allowed here; instead `git reset --soft HEAD~1`, repair, re-verify Task 3, and re-commit (the single-commit rule).

- [ ] **Step 4: Clean up the snapshot**

```bash
rm /tmp/findings-pre-split.md
```
