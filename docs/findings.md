# LCE — Hallazgos (Findings)

**Última actualización:** 2026-06-10
**Estado del proyecto:** Fase 1 (fundación) y Fase 2 (pipeline completo) fusionadas en `main`. Fase 3 (compresión Markdown outline-only y endurecimiento del benchmark) completada en la rama `lce-phase3-compression`.
**Entorno:** EndeavourOS, i5-13420H, 16 GiB RAM, RTX 3050 6 GiB VRAM, CUDA 13.3,
llama-cpp-python 0.3.28 (cuBLAS), Qwen2.5-3B-Instruct Q4_K_M, ChromaDB 1.5.9 (embeddings CPU).

---

## 1. Resultados del primer benchmark honesto

Batería de 15 consultas × 3 brazos, reps=1, corpus = este repositorio (30 archivos `.py`/`.md`):

| brazo | prompt_tokens (media) | ahorro % | TTFT ms | total ms | format_success |
|---|---|---|---|---|---|
| naive | 935.5 | 0.0 | 431.0 | 831.6 | 1.00 |
| lean | 660.3 | **29.4** | 293.3 | 727.3 | **0.93** |
| lean_grammar | 660.3 | **29.4** | 302.5 | 1095.5 | **1.00** |

Lecturas principales:

- **El ahorro de tokens es real pero está lejos del objetivo (>80%).** Ver §3.
- **La gramática GBNF aporta valor medible desde el primer día:** el brazo lean
  sin gramática falló el formato en 1 de 15 consultas (0.93); con gramática el
  fallo es imposible a nivel de sampler (1.00). Ese 7% de fallos es exactamente
  el coste que la decodificación restringida elimina.
- **La gramática tiene un coste de latencia:** ~+370 ms de generación total
  frente al brazo lean (1095 vs 727 ms). El TTFT no se ve afectado (~300 ms en
  ambos brazos lean). Trade-off honesto: formato garantizado a cambio de
  generación más lenta por token.
- **TTFT escala con el tamaño del prompt:** 431 ms (naive, ~935 tokens) frente a
  ~300 ms (lean, ~660 tokens). La poda de contexto se traduce directamente en
  latencia inicial menor — el mecanismo central de la hipótesis del proyecto.
- El objetivo de **TTFT sub-segundo en hardware de consumo se cumple** en los
  tres brazos.

## 2. Hallazgo metodológico: el caché de prefijos de llama.cpp sesgaba el TTFT ~10×

El primer run del benchmark reportó TTFT de **25.9 ms** para `lean_grammar`
frente a 272 ms para `lean` — con prompts idénticos. La causa no fue la
gramática: `Llama.generate()` en llama-cpp-python reutiliza el estado KV para
prefijos de prompt compartidos **dentro de una misma instancia** (lógica
`longest_prefix`, `llama.py:905-937`), independientemente del `LlamaCache`
(que está deshabilitado por defecto). Como los brazos corren consecutivamente
con prompts idénticos o solapados, el orden de ejecución contaminaba la métrica.

**Corrección (commit `1778578`):** `Engine.generate` resetea el contexto antes
de cada generación, de modo que cada transacción paga la evaluación completa de
su prompt. El TTFT pasa a ser proporcional al tamaño del prompt y comparable
entre brazos y repeticiones. Tras el fix: 302.5 ms (coherente con los 293.3 ms
de lean).

Moraleja para cualquiera que haga benchmarks con llama-cpp-python: revisar
`self.cache` no basta; el prefix-matching interno existe siempre.

## 3. La compresión semántica depende fuertemente del tipo de documento

Medido sobre los archivos reales del repositorio:

| tipo | reducción de tamaño (esqueleto vs crudo) |
|---|---|
| Python (esqueleto AST: firmas + primera línea de docstring) | **78–86%** |
| Markdown (título + encabezados + resúmenes de sección) | **~21%** |

El corpus actual es pesado en documentación (specs y planes largos), lo que
arrastra el ahorro global al 29.4%. **El objetivo >80% es alcanzable para
código, no (todavía) para prosa.** Palancas identificadas para la fase 3:
mejorar la compresión de esqueletos Markdown, ponderar el corpus hacia código,
o reportar el ahorro segmentado por tipo de documento.

## 4. Notas estadísticas (validez de las métricas)

- `prompt_tokens` es determinista dado un índice fijo: las repeticiones no
  añaden muestra al ahorro (n efectivo = 15 por brazo), solo a las métricas de
  generación (TTFT, latencia, format_success).
- La generación no fija semilla: `format_success_rate` es una estimación
  puntual muestreada, no reproducible bit a bit entre runs.
- n=15 es escala de humo. Antes de publicar cifras titulares: ampliar batería
  y repeticiones.

## 5. Gotchas de ingeniería encontrados (y corregidos) por el proceso de revisión

Cada uno fue detectado por revisión de código por agentes y corregido con test de regresión:

1. **Off-by-one en completion_tokens** — `create_completion(stream=True)`
   siempre emite un chunk centinela final con `finish_reason` y texto vacío;
   contarlo infla la métrica en +1 (`f0b7e00`).
2. **Paridad de tokenización** — `tokenize()` usa `special=False` por defecto,
   pero `create_completion` tokeniza con `special=True`; sin igualarlo, el
   conteo de prompt_tokens difiere de lo que el modelo evalúa (relevante con
   plantillas ChatML) (`f0b7e00`).
3. **Caracteres de control en GBNF** — `[^"\\]` admite control chars crudos que
   `json.loads` rechaza; usar `[^"\\\x7F\x00-\x1F]` (mismo bug histórico del
   json.gbnf de llama.cpp) (`f96a8a2`).
4. **Chunks fantasma en re-indexado** — upsert por id no borra chunks antiguos
   si un documento encoge; hay que borrar por `source_id` antes de re-insertar
   (`7ea244b`).
5. **Encabezados falsos en Markdown** — líneas `#` dentro de bloques de código
   cercados generaban ~56% de ruido en los esqueletos de los propios docs del
   repo (`92bee68`).
6. **Símbolos condicionales perdidos** — funciones/clases definidas dentro de
   `if`/`try`/`for` desaparecían del esqueleto AST (`1932f70`).
7. **Conexiones SQLite sin cerrar** — `with sqlite3.connect(...)` solo gestiona
   la transacción, no el cierre (`019a182`).
8. **Sesgo de caché de prefijos en TTFT** — ver §2 (`1778578`).

## 6. Verificación

- Plantilla ChatML verificada byte a byte contra el `tokenizer.chat_template`
  embebido en el GGUF; `<|im_end|>` (token 151645) es EOS y termina la
  generación de forma natural en los tres brazos.
- Suite: 57 tests unitarios (sin GPU) + 3 tests GPU (smoke, e2e tres brazos,
  batería completa). Todo en verde sobre `main`.

## 7. Próximos pasos candidatos (fase 3)

- Compresión de esqueletos Markdown (la palanca principal hacia el >80%).
- Ahorro segmentado por tipo de documento en la tabla del benchmark.
- Opción de semilla fija para runs reproducibles.
- Ampliar batería (>15 consultas) y reps antes de publicar cifras titulares.
- UX: `lce index <ruta inexistente>` reporta "indexed 0 files" con exit 0 en
  lugar de error.

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
