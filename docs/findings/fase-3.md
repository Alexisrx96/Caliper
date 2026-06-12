> Fase 3 (2026-06-10) — §8, movido íntegro desde `docs/findings.md` el 2026-06-12.
> Publicado en `bda3bff`; historia previa: `git log -- docs/findings.md`.

## 8. Fase 3 — Compresión markdown outline-only y benchmark endurecido (2026-06-10)

Cambios medidos en esta fase: los esqueletos Markdown se reducen a outline puro
(título + jerarquía de encabezados, **sin resúmenes de sección**); la batería se
amplió de 15 a 30 consultas (10 navegación / 10 búsqueda / 10 explicación); el
run es sembrado (`--seed 42`; la repetición *i* usa `seed+i`) e índice
reconstruido desde cero (`--reindex`). Corpus = este repositorio (**33 archivos,
0 omitidos**), reps=3 → **270 transacciones**. Suites en verde antes del run:
unitarios `68 passed, 3 deselected`; GPU `3 passed, 68 deselected`.

### Tabla por brazo (270 transacciones)

| brazo | n | prompt_tokens (media) | ahorro % | TTFT ms | p50 | total ms | format_success |
|---|---|---|---|---|---|---|---|
| naive | 90 | 956.9 | 0.0 | 462.0 | 454.8 | 908.5 | 1.00 |
| lean | 90 | 540.5 | **43.5** | 250.0 | 236.3 | 696.9 | **1.00** |
| lean_grammar | 90 | 540.5 | **43.5** | 292.6 | 278.9 | 2190.4 | **1.00** |

### Compresión del corpus por tipo (chars, crudo vs esqueleto)

| kind | raw_chars | skeleton | reducción % |
|---|---|---|---|
| code | 67861 | 10500 | **84.5** |
| doc | 139416 | 3097 | **97.8** |
| overall | 207277 | 13597 | **93.4** |

### Comparación con la línea base de fase 2

| métrica | fase 2 (n=15, reps=1) | fase 3 (n=30, reps=3) |
|---|---|---|
| ahorro prompt_tokens (lean vs naive) | 29.4% | **43.5%** |
| reducción esqueleto doc (chars) | ~21% | **97.8%** |
| reducción esqueleto code (chars) | 78–86% | **84.5%** |
| reducción global (chars) | (no medido) | **93.4%** |
| format_success brazo lean | 0.93 | **1.00** |

El objetivo **>80% de compresión se cumple en chars**: globalmente (93.4%) y por
tipo (code 84.5%, doc 97.8% — la palanca outline-only multiplicó por ~4.7 la
reducción de docs desde ~21%). En cambio, **el ahorro de prompt_tokens medido por
telemetría sube de 29.4% a 43.5% pero NO alcanza el 80%**: son métricas distintas
— el prompt incluye el scaffolding fijo (sistema + plantilla ChatML + la consulta)
y el payload de los k=3 documentos recuperados, así que la compresión del corpus
no se traslada 1:1 al prompt. `format_success` es 1.00 en los tres brazos (la fase
2 medía 0.93 en lean sin gramática; con n=90 por brazo este run no registró ningún
fallo de formato — nota: es una estimación muestral, no una garantía, salvo en el
brazo con gramática donde el formato es estructural). El TTFT sigue escalando con
el tamaño del prompt: 462 ms naive vs 250 ms lean. El gap de latencia total del
brazo con gramática se **amplió notablemente**: 2190 ms vs 697 ms del lean (en fase
2 era 1095 vs 727); flag honesto: vale la pena investigarlo en una fase futura
(posible interacción gramática×semilla o coste por token de la decodificación
restringida con las consultas nuevas). El run es reproducible:
`uv run lce bench --seed 42 --reindex`.

**Actualización 2026-06-11:** la especulación gramática×semilla queda
descartada — es coste por token estructural, amplificado porque este run se
ejecutó en batería; ver §9.

> **Actualización fase 5:** ver §10 para el barrido de `k`/`doc_cap` y el
> veredicto sobre el objetivo de >80% de ahorro.

