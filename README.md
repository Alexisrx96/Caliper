# Lean Context Engine (LCE)

Experimento de "Coding in Public": eficiencia de tokens y harnessing estricto
de SLMs locales.

## Objetivos
- **Ahorro de contexto:** poda semántica (esqueletos AST + metadatos) con
  objetivo de >80% menos prompt tokens vs. Naive RAG.
- **Cero alucinaciones de formato:** decodificación restringida (GBNF) con
  esquema de enrutamiento JSON.
- **Hardware de consumo:** TTFT sub-segundo en 16GB RAM / 6GB VRAM.

## Baseline de hardware
- EndeavourOS (Arch) + i3 v4.25.1
- Intel Core i5-13420H (8 núcleos), 16 GiB RAM
- NVIDIA RTX 3050 Laptop, 6144 MiB VRAM
- Driver NVIDIA v610.43.02, CUDA UMD v13.3

## Metodología
Cada transacción se registra en `experiment_logs.db` (SQLite, WAL):
`prompt_tokens`, `completion_tokens`, `ttft_ms`, `total_latency_ms`,
`format_success`. Tres brazos de benchmark: **naive** (chunks crudos),
**lean** (esqueletos, sin gramática) y **lean_grammar** (esqueletos + GBNF).

## Uso (CLI)
```bash
uv run lce index .                      # indexa .py/.md en colecciones raw + skeleton
uv run lce ask "¿dónde está el esquema de telemetría?"          # lean + GBNF (default)
uv run lce ask "..." --no-grammar       # brazo lean (solo prompt)
uv run lce ask "..." --mode naive       # brazo naive (chunks crudos)
uv run lce bench                        # batería 15 consultas × 3 brazos × 3 reps
```
`lce ask` imprime el JSON de enrutamiento por stdout y la línea de métricas
(tokens, TTFT, total) por stderr — componible con pipes (`| jq .action`).

## Replicar
1. Linux con CUDA, ≥16GB RAM, ≥6GB VRAM.
2. `./scripts/setup_env.sh` — compila llama-cpp-python (cuBLAS) y descarga
   Qwen2.5-3B-Instruct Q4_K_M.
3. `uv run pytest` (unit) · `uv run pytest -m gpu` (smoke/e2e/benchmark).

Diseño: `docs/superpowers/specs/2026-06-09-lce-foundation-design.md` (fase 1)
y `docs/superpowers/specs/2026-06-09-lce-phase2-pipeline-design.md` (fase 2).
