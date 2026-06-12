> Fase 4 (2026-06-11) — §9, movido íntegro desde `docs/findings.md` el 2026-06-12.
> Publicado en `9298540`; historia previa: `git log -- docs/findings.md`.

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

