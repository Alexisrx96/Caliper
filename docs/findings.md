# LCE — Hallazgos (Findings)

**Última actualización:** 2026-06-11
**Estado del proyecto:** Fases 1–3 fusionadas en `main`. Fase 4 (benchmarking honesto + decodificación restringida barata) completada en la rama `lce-phase4-honest-bench`.
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

**Actualización 2026-06-11:** la especulación gramática×semilla queda
descartada — es coste por token estructural, amplificado porque este run se
ejecutó en batería; ver §9.

## 9. Fase 4 — Causa raíz de la latencia con gramática y benchmarking honesto (2026-06-11)

### 9.1 Causa raíz: el sampler de gramática enmascara el vocabulario completo en cada token

llama-cpp-python 0.3.28 añade el sampler GBNF **antes** de top-k en la cadena
(`llama.py:747` vs `:773`), así que cada token generado paga una validación de
gramática sobre los ~152k tokens del vocabulario de Qwen: **~+30 ms/token en
CA** (medido: 46.1 vs 15.7 ms/token, ~3× el decode). La semilla es inocente:
un experimento controlado 2×2 (gramática on/off × semilla on/off, cada token
cronometrado) mostró coste por token idéntico con y sin semilla, y salidas
idénticas bajo la misma semilla.

### 9.2 El run de §8 se ejecutó en batería (y nada lo registraba)

El historial de upower (`/var/lib/upower`) prueba que el benchmark de fase 3
(`bench-20260610-230027`, 23:00 local) corrió **enteramente en batería**:
descarga de 22:30 a 23:25 con 71–91 W durante el run. Eso infló el
enmascarado CPU de la gramática a ~53 ms/token (~1.8× el coste en CA). Además,
el mismo prompt de ~420 tokens midió TTFT de **55 ms** (máquina descansada,
CA), **~187 ms** (carga sostenida en CA — el presupuesto de boost se agota a
los ~30–60 s) y **~250–510 ms** (batería): el estado de la máquina mueve
componentes individuales hasta ~5× y ninguna cifra previa lo registraba.

**Corrección metodológica (esta fase):** cada transacción guarda un snapshot
de estado (`machine_state` JSON: CA/batería, governor y frecuencia de CPU,
pstate/clocks/temperatura/potencia de GPU); `lce bench` **rehúsa arrancar en
batería** (`BenchOnBatteryError`, exit 2; `--allow-battery` para forzar) y la
tabla imprime un pie con las transacciones contaminadas y el rango de clocks.

### 9.3 La solución: muestrear-y-validar (y por qué no bastaba reordenar)

El primer intento (mover la gramática después de top-k: máscara sobre ≤40
candidatos) medía ~0 overhead pero **aborta el proceso en C** cuando ningún
candidato superviviente es válido: el sampler dist elige un token arbitrario
del array enmascarado y el accept de la gramática lanza
`std::runtime_error: "Unexpected empty grammar stack after accepting piece"`
→ SIGABRT, incapturable desde Python (~1/30 pasadas de batería sin semilla;
con semilla 42 pasaba — falsa confianza). La solución definitiva porta la
estrategia del propio llama.cpp (`common/sampling.cpp`, `grammar_first=false`),
`lce/sampling.py`:

1. Muestrear con la cadena normal **sin gramática** (mismo coste y mismo
   stream RNG que el brazo lean).
2. Validar **solo el token muestreado** contra un sampler de gramática
   independiente (un paseo por el texto del token, ~µs).
3. Si se rechaza (*rescate*): enmascarar el vocabulario **completo** con la
   gramática — nunca queda vacío — y remuestrear con la cadena. La columna
   `grammar_fallback` registra los rescates.

Test de regresión GPU: la batería completa de 30 consultas sin semilla (la
carga que reproducía el SIGABRT) atraviesa el brazo con gramática sin abortar
y con todas las salidas válidas.

### 9.4 Antes/después controlado (mismo día, CA, máquina descansada, seed 42)

Corpus = este repositorio (**39 archivos** — creció con el código de fase 4;
los prompt_tokens no son comparables con §8), reps=3, 270 transacciones por
run, índice reconstruido (`--reindex`) en el run "antes" y reutilizado en el
"después". Pie de máquina en ambos: `battery transactions 0/270`; clocks GPU
1402–2002 MHz.

**Antes** (`--grammar-first`, cadena gramática-primero de fase ≤3):

| brazo | prompt_tokens | ahorro % | TTFT ms | total ms | ms/token gen | fmt_ok |
|---|---|---|---|---|---|---|
| naive | 949.5 | 0.0 | 442.9 | 907.5 | 15.8 | 0.93 |
| lean | 643.7 | 32.2 | 291.7 | 717.1 | 15.7 | 1.00 |
| lean_grammar | 643.7 | 32.2 | 317.2 | **1586.5** | **46.1** | 1.00 |

**Después** (muestrear-y-validar):

| brazo | prompt_tokens | ahorro % | TTFT ms | total ms | ms/token gen | fmt_ok |
|---|---|---|---|---|---|---|
| naive | 949.5 | 0.0 | 444.4 | 908.9 | 15.8 | 0.93 |
| lean | 643.7 | 32.2 | 294.8 | 721.7 | 15.7 | 1.00 |
| lean_grammar | 643.7 | 32.2 | 291.6 | **716.3** | **15.6** | 1.00 |

Titular: **la penalización de la gramática desaparece** — 1586.5 → 716.3 ms
de media total (−55%), por token 46.1 → 15.6 ms (= lean), TTFT incluido.
Rescates: **0/270** (el modelo, muestreado sin restricción, produjo JSON
válido en todas las posiciones; la gramática queda como garantía estructural
gratuita). Los brazos naive/lean reproducen entre runs dentro de ~1% — la
evidencia de que el protocolo controlado funciona. El 0.93 de fmt_ok en
naive son fallos de formato muestreados del brazo sin gramática con prompt
grande (determinista bajo seed 42, idéntico en ambos runs).

Reproducción: `uv run lce bench --seed 42 --reindex --grammar-first
--run-id phase4-before` y `uv run lce bench --seed 42 --run-id phase4-after`.

### 9.5 Lecciones de metodología

1. **Registrar el estado de la máquina o no publicar latencias.** Las cifras
   de §1 y §8 mezclan estados (batería, boost, sostenido) sin saberlo; las
   comparaciones entre fases eran arena movediza.
2. **Las pruebas con semilla dan falsa confianza ante fallos dependientes del
   muestreo**: el aborto del reordenado post-top-k solo aparecía sin semilla.
3. **`pytest | tail` enmascara un SIGABRT**: el exit code del pipeline es el
   de `tail`. Comprobar `PIPESTATUS` o no entubar.
4. **El coste de la decodificación restringida no es inherente**: es una
   decisión de orden de samplers. Validar el token muestreado (estrategia del
   propio llama.cpp) da la garantía estructural a coste ~cero.
