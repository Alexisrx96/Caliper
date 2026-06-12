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
