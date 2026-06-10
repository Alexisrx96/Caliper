# LCE — Hallazgos (Findings)

**Última actualización:** 2026-06-10
**Estado del proyecto:** Fase 1 (fundación) y Fase 2 (pipeline completo) fusionadas en `main`.
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
