> Fase 5 (2026-06-11) — §10, movido íntegro desde `docs/findings.md` el 2026-06-12.
> Publicado en `1d25b1b` (ajuste `d0fe6f3`); historia previa: `git log -- docs/findings.md`.

## 10. Fase 5 — Ajuste de ahorro de prompt bajo restricción de calidad (2026-06-11)

Pregunta de la fase: ¿cuánto ahorro de tokens de prompt se puede extraer
(con `k`, `doc_cap` y un system prompt recortado) **sin perder calidad de
ruteo**? Hasta ahora ningún recorte de payload tenía costo visible:
`format_success` se mantiene en 1.00 aunque el router elija targets
equivocados. La fase añade primero la métrica que hace visible ese costo y
después barre las palancas bajo ella.

### 10.1 La métrica: target hit-rate

Cada una de las 30 consultas de la batería lleva ahora un regex de target
esperado (`BatteryQuery(query, expect_target)`, compilado en import con
IGNORECASE). `target_hit(text, expect)` puntúa 1 si la respuesta es JSON de
ruteo válido **y** el regex matchea el campo `"target"` (vía `re.search`);
un fallo de formato es automáticamente un miss. Se almacena por transacción
(columna `target_hit`, NULL en filas previas a la fase 5) y se agrega por
brazo como `target_hit_rate` (columna `hit` en la tabla del bench).

### 10.2 La brecha estructural compresión→ahorro

El corpus comprime **94.0%** en caracteres (la cifra de fase 4 era 93.7%
sobre 39 archivos; misma métrica, corpus de 41) pero el ahorro de
prompt intra-run ronda el 30–35%. Dos hechos estructurales lo explican:

1. **La compresión del corpus no llega al prompt tal como se compara**: el
   brazo naive recupera *chunks* (~280 tokens promedio por doc), el lean
   *esqueletos completos* (~179) — solo ~36% más pequeños por documento. El
   94% compara archivos completos; el prompt nunca contiene archivos
   completos.
2. **El piso fijo de andamiaje**: system prompt + marcadores ChatML. El
   system prompt pasó de 83 a **48 tokens** (medido con el tokenizer de
   qwen2.5, `vocab_only`), recorte aplicado a todos los brazos y configs —
   el andamiaje fijo baja de ~98 a ~63 tokens.

Corolario que el barrido confirma: reducir `k` encoge naive y lean a la
vez, así que el ahorro *intra-run* apenas se mueve; la palanca solo luce en
la comparación *entre* configs (lean ajustado vs naive por defecto, §10.5).

### 10.3 El barrido (mismo día, CA, descansada, seed 42, 270 transacciones/config)

Cinco configs, una sola indexación (`--reindex` solo en el baseline; el cap
es prompt-time). Batería 0/270 en todos los runs; clocks GPU 1380–2010 MHz.
Columnas: tokens de prompt lean (media), ahorro intra-run lean, hits del
brazo lean_grammar (de 90), TTFT y total del brazo lean_grammar,
format_success del brazo naive.

| config | lean tok | ahorro intra-run | hits /90 | TTFT ms | total ms | fmt naive |
|---|---|---|---|---|---|---|
| `p5-baseline` (k=3) | 651.8 | 32.7% | **64** | 291.3 | 740.3 | 0.82 |
| `p5-k2` | 451.8 | 30.0% | 59 | 202.4 | 678.7 | 0.86 |
| `p5-k1` | 226.5 | 34.8% | 58 | 102.1 | 506.5 | 0.96 |
| `p5-k3-cap30` | 643.6 | 19.4% | 62 | 290.2 | 734.0 | 0.87 |
| `p5-k1-cap30` | 230.1 | 21.3% | 58 | 104.3 | 510.7 | 0.98 |

Salida cruda completa: `docs/superpowers/plans/p5-sweep-results.txt`.

### 10.4 Regla de decisión: ningún candidato pasa

Barra mecánica: pasa el candidato cuyo brazo lean_grammar pierda ≤1 hit
frente al baseline (64/90). Resultado: **ninguno pasa** — `p5-k3-cap30`
pierde 2; `p5-k2` pierde 5; `p5-k1` y `p5-k1-cap30` pierden 6. Los
**defaults quedan en `k=3`, sin cap** (sin cambios de código).

El déficit es señal, no ruido: por rep (semillas 42/43/44, hits de 30) el
baseline da 21/22/21 y los recortes de `k` pierden en las tres semillas
(`p5-k1`: 22/18/18, `p5-k2`: 19/20/20). El slot adaptativo del protocolo se
omitió: la frontera es clara — recortar documentos cuesta 5–6 hits de forma
consistente y el cap no puede recuperar información de documentos que ya no
se recuperan (`k2+cap30` estaba dominado por `k2`, que ya falla por 5).

### 10.5 Veredicto sobre el >80%

**No se alcanza >80% de ahorro sin pérdida de calidad.**

- *A paridad de calidad* (la única config que pasa su propia barra es el
  baseline): el ahorro honesto es **32.7%** intra-run.
- *Comparación de despliegue* (lean ajustado k=1 vs naive por defecto k=3,
  entre configs): 226.5 vs 967.8 tokens = **76.6%** — clavado en la banda
  que la aritmética del spec predecía (190–230 tokens), pero por debajo de
  80% y con un costo de 6/90 hits que la regla de decisión rechaza.

El techo es estructural en este corpus: el andamiaje fijo (~63 tokens) más
un esqueleto (~179) ya suman ~240 contra un naive k=1 de ~347. Empujar más
allá exige comprimir el esqueleto por documento o cambiar la composición
del prompt, no ajustar `k`/cap.

### 10.6 Hallazgos colaterales

- **Lean le gana a naive en calidad de ruteo a k=3: 64 vs 55 hits.** Los
  chunks crudos saturan al router 3B — `format_success` naive cae a 0.82 y
  cada fallo de formato es un miss. Con prompts chicos naive recupera el
  formato (0.96–0.98 a k=1). A k=3, lean no solo es 32.7% más barato y ~30%
  más rápido: es *más preciso*.
- **El corpus se contaminó con la documentación del propio experimento**:
  39→41 archivos (spec y plan de fase 5 indexados). Para la consulta del
  cómputo de ahorro, los dos primeros vecinos son ahora esos documentos
  (near-tie, Δdistancia 0.005) en lugar de `lce/bench.py`.
- **Inestabilidad de retrieval entre procesos con k=1**: ese near-tie hizo
  flip del doc top-1 entre dos runs de la misma config (1/30 consultas,
  prompt de 203→311 tokens). Con k=3 el efecto se diluye; con k=1 cada
  flip cambia el prompt completo.
- **Comparabilidad**: los números absolutos de fase 4 ya no son comparables
  — system prompt recortado (todos los brazos) y corpus de 41 archivos.

### 10.7 Verificación

- `uv run pytest`: 115 passed (4 GPU deseleccionados).
- `uv run pytest -m gpu`: 4 passed sobre el modelo real — la consulta del
  esquema de telemetría puntúa `target_hit == 1` en los tres brazos.
- Barrido: 5 runs con exit 0, batería 0/270 en todos.

Reproducción: `uv run lce bench --run-id p5-baseline --seed 42 --reindex` y
después `--k`/`--doc-cap` según la tabla (sin `--reindex`).
