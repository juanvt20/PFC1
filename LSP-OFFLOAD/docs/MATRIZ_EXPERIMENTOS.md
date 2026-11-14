# Matriz de experimentos, métricas y evidencia

Estado: revisión documental del 13 de noviembre de 2026. No se han preparado
ni ejecutado nuevos experimentos en esta revisión. No hay resultados propios
de entrenamiento. La existencia de código, la validación de una medición
y la viabilidad de la comparación completa son estados distintos.

Fuente: `referencias/LSP-OFFLOAD.pdf`, páginas del PDF desde 1; upstream
`d9dc92a32b7e4829116ba6abec17c280dfb548a4`. Las rutas de código siguientes
son relativas a `original/upstream/`. Se revisaron de nuevo visualmente las
páginas 3, 6, 7, 11 y 12 para comprobar títulos, ejes, unidades y leyendas.
Las diferencias Dxx/Nxx remiten a [CONTRASTE_PDF_CODIGO.md](CONTRASTE_PDF_CODIGO.md)
y [COMPATIBILIDAD.md](COMPATIBILIDAD.md).

## Alcance y estados

Hardware comprobado: RTX 5050 Laptop, 7.96 GiB de VRAM total y 6.90 GiB
libres en la instantánea; i7-13700HX; Linux ve 7.61 GiB de RAM, con
6.22 GiB disponibles en la última instantánea. No son reservas garantizadas
para una futura ejecución. Fuente: `configs/environment_validation.json`.

- **Candidato**: presupuesto razonable para comenzar con una prueba pequeña;
  todavía no se ha demostrado el entrenamiento ni el pico de memoria.
- **Condicional**: requiere comprobar memoria o completar información;
  no se admite como experimento ya viable.
- **No viable en la ruta original**: existe una cota de memoria que excede
  la capacidad; no se sustituye por cuantización, recorte o cambio de modelo.
- **Implementación parcial**: hay funciones upstream, pero faltan partes
  del protocolo o del registro. Ningún experimento del paper está validado
  localmente. Solo los diagnósticos de entorno de la última tabla se ejecutaron.

N15 (reinicio de M/V tras su transporte), N01 (epsilon CPU/GPU), las
diferencias del compresor y la corrección del planificador siguen abiertos.
Esta matriz no autoriza ni aplica cambios matemáticos. La viabilidad de
LSP no implica que quepan sus líneas base; un baseline OOM se registra como
inviable para esa configuración, no como una medición de velocidad cero.

En las tablas, `R` significa `results/<run_id>/`, con un directorio por
modelo, tarea, método y semilla. Los destinos bajo `R/paper/` y
`R/diagnostics/` son **contratos de salida propuestos, aún sin escritor**.
No se han creado CSV vacíos que puedan confundirse con resultados.
Los archivos upstream indicados son salidas que el código sabe generar,
no evidencia de que se hayan generado aquí. `result_ref/` contiene datos
de los autores y no se usará como medición local.

## Matriz de métricas originales y candidatos

Los protocolos P1-P6, detallados más abajo, forman parte de cada fila.
Las métricas auxiliares no sustituyen la columna de métrica original.

| ID y experimento | Tabla/figura, página y sección | Métrica original y unidad | Línea base original | Protocolo de evaluación | Archivo que recogerá la medición | Implementación y viabilidad local |
| --- | --- | --- | --- | --- | --- | --- |
| E01. RoBERTa-base, calidad GLUE con presupuesto de 1 h | Tabla 3, p.6, “Accuracy validation ... GLUE”; p.11, “GLUE Experiment” | Puntuación por tarea, rotulada “accuracy”; escala fraccionaria, sin multiplicar por100. `Avg` publicado tiene ambigüedad de tareas | Full Parameter y GaLore rango16; LSP d512,r16 | P1: batch16, LR5e-5, límite1h, siete tareas identificadas; no reconstruir la octava columna SST2 repetida ni atribuir al paper una selección de checkpoint no indicada | Existente: `<output>/eval_losses.csv`, columna `glue`. Propuestos: `R/paper/glue_evaluations.csv` con todos los campos y `R/paper/glue_budget.csv` con checkpoint/paso/presupuesto | **Parcial, candidato**. `calculate_glue` y `Trainer.train_epoch` existen; `run_glue.sh` activa solo MNLI y no lanza GaLore. Faltan persistir métricas desglosadas, siete tareas, baseline GaLore verificado y política de checkpoint. MRPC/SST2 corto será solo diagnóstico |
| E02. Convergencia GLUE por iteración | Fig.8, p.11, “GLUE Experiment” | `Eval Acc` frente a `Iterations`: puntuación por tarea y número de iteraciones, no horas ni tokens | Full Param frente a LSP; la figura no incluye GaLore | P1: mismos splits/evaluador de E01; guardar iteración global y época; no confundir `iter` local de época con contador global | Existente: `eval_losses.csv` con `epoch,iter,time,glue`. Propuesto: `R/paper/glue_evaluations.csv`, campos `global_step,epoch,iter,metric,value` | **Parcial, candidato**. Hay evaluación cada100 iteraciones en script; faltan contador global persistido, desglose y exportador de curvas. No prueba offloading CPU: script usa `cuda:0` |
| E03. GPT2-774M, Alpaca | Fig.5a, p.7, “Comparison with Zero-Offload / PEFT”; configuración p.11 | Perplexity de evaluación, adimensional, menor mejor; eje tiempo en h. Tiempo para alcanzar la misma perplexity, h; reducción de tiempo, % | Zero-Offload y LoRA (figura: rango16); LSP d512,r8/16/32 | P2: datos/tokenización/split fijados, batch4, longitud512 del script; preservar definición upstream de ppl. P3 para tiempo real. No usar 33.1%-62.5% como objetivo esperado local | Existentes: `eval_losses.csv` (`ppl`) y `losses.csv`. Propuestos: `R/paper/alpaca_evaluations.csv`, `R/paper/time_to_quality.csv`; tiempos brutos en `R/diagnostics/timeline.csv` | **Parcial; LSP/LoRA candidatos, comparación Zero condicional por RAM**. Faltan reloj íntegro, selección registrada de checkpoint, resolución de LR/rango/umbral y trazas del scheduler. No se ha probado batch4 ni una recalibración de modelo |
| E04. Ablación de throughput en GPT2-774M | Fig.6, p.7, “Ablation Study” | Iteraciones de entrenamiento/s; figura: throughput normalizado, cociente sin unidad con Zero=1 | Zero; Zero + Layerwise Schedule; LSP (d2048,r16), (d1024,r16), (d512,r16), (d256,r16); referencia GPU FWD+BWD+UPD | P3: igual modelo, batch, longitud y precisión entre rutas; distinguir microbatch de actualización; completar trabajo pendiente en límites de ventana; medir referencia GPU con UPD real. Validar cada d, no asumir que el mayor cabe | Parcial existente: stdout de `para_offload_modules.py` (`average running time`). Propuestos: `R/paper/throughput.csv` y `R/paper/normalized_throughput.csv`, con ID de baseline | **Parcial, condicional**. Existen modos de scheduler y temporización sintética, no matriz automatizada ni denominador GPU validado. Si Zero no cabe, no se publica el cociente. Cómputo aislado no demuestra entrenamiento full-rank residente en GPU |
| E05. Desglose de Zero en GPT2-774M | Fig.2, p.3, “Case Study on Zero’s Schedule” | Slowdown normalizado respecto al cómputo GPU, sin unidad; segmentos GPU, Comm, CPU, Other como contribuciones no solapadas | Cómputo GPU de referencia=1; Zero para la misma carga | P3: fp16 y gradient checkpointing como caption; batch4 en la barra original GPT2-774M. Separar perfil de esa barra de búsqueda local de batch máximo. Trazas para atribuir tiempo exclusivo; no sumar duraciones solapadas | Existente: stdout de `Trainer.offload_profile` / `Scheduler.report`. Propuestos: `R/paper/slowdown.csv`, `R/paper/critical_path.csv`, `R/trace/` | **Parcial, condicional a Zero/RAM y trazas**. Faltan validar denominador, desglose exclusivo y exportación. GPU-only `PROFILE_COMPUTE` no acredita por sí solo FWD+BWD+UPD. No confundir este slowdown con el throughput normalizado de E04 |
| E06. Perfil offload DeepSeek-Coder-1.3B | Fig.7a, p.7, “Training time breakdown”; p.12, “Instruction Fine-tuning with DeepSeek-Coder” | Latencia por iteración, s; segmentos GPU Compute, Comm, CPU, Other, s | Zero-Offload frente a LSP d1280,r4 | P3: perfil publicado batch1×384 tokens, distinto del batch efectivo128×1024 de calidad. README usa datos sintéticos, warmup5 y repetición10; comando Zero omite longitud384 y hereda1024: explicitar el perfil común antes de comparar | Existente: stdout/NVTX de `para_offload_modules.py:634–796`. Propuestos: `R/paper/iteration_latency.csv`, `R/paper/critical_path.csv`, `R/trace/` | **Parcial, condicional**. Pesos caben, buffers y Zero pueden no caber en RAM. Falta exportar tiempos y validar límites/trazas; el fit inicial sin aprendizaje no valida calidad de proyectores |
| E07. Error de proyectores DeepSeek-Coder-1.3B | Fig.7b, p.7, “Hyperparameter”; Fig.9a/b, p.12, “LSP-Offload’s Hyperparameters” | Error relativo de estimación α, adimensional, en calibración y validación; eje r (2,4,...,512), familias d512/768/1024/1280 | Proyectores ortogonales SVD de GaLore, mismo r; no es loss de entrenamiento | P4: gradientes de conjuntos separados, mismo modelo/checkpoint y módulos; guardar razón de normas por matriz y agregación, sin confundir con el objetivo regularizado del paper | Existente: `BenchmarkContext.benchmark` devuelve dos escalares; `Dummy.step` puede guardar `.pt`. Propuesto: `R/paper/projector_error.csv` con split, módulo, d, r, método, error y agregado identificado | **Parcial, condicional por RAM y segunda ruta**. No hay lanzador completo del barrido ni CSV; no se ha probado el import de esta ruta. Reducir módulos/muestras sería estudio parcial rotulado, no reproducción íntegra de Fig.9 |
| E08. Calidad por pasos DeepSeek-Coder-1.3B | Fig.5c, p.7; simulación descrita p.12 | `Train Loss`, adimensional; figura original frente a h **simuladas**. Pérdida por pasos es dato de entrada a esa simulación | Full Parameter como calidad de Zero; LoRA rango8, alpha32; GaLore rango256, alpha0.25; LSP d1280,r4 | P5: batch efectivo128, longitud1024, AdamW, cosine, hasta5 épocas. Calidad en GPU suficiente + perfil offload escalado son fases distintas; no prometer aquí esa ruta GPU completa | Existente LSP: `<output>/runs/events.out.tfevents.*`, tag `loss/train`; baselines HF: estado/logs Trainer. Propuesto: `R/paper/code_train_loss.csv`. Reconstrucción eventual: `R/simulation/quality_vs_estimated_time.csv` | **Parcial, diferido/condicional**. Segunda ruta no validada y memoria de todos los baselines sin probar. Offload real reducido sería adaptación. No generar aún ejes120h ni declararlos wall-clock |
| E09. HumanEval multilenguaje, DeepSeek-1.3B | Tabla 4 superior, p.6; p.12, “Instruction Fine-tuning with DeepSeek-Coder” | `Evaluation accuracy`: puntuaciones en escala tabulada0–100 para python/java/cpp/js/ts/php y `Avg`; **no identificada como pass@k**. `GPU Mem` en GB; `Time` en h equivalentes simuladas | Zero, LoRA rango8, GaLore rango256 y LSP d1280,r4; checkpoint equivalente a ~120h | P6: protocolo incompleto. No calcular score comparable sin identificar harness, dataset, generación y evaluación. Definición exacta de `GPU Mem` también pendiente | No hay evaluador HumanEval localizado. Propuestos y bloqueados: `R/paper/humaneval_scores.csv`, `humaneval_protocol.json`, `reported_memory.csv`; generaciones/resultados brutos separados | **Falta implementación y protocolo; hardware de inferencia condicional**. No es imposibilidad demostrada de1.3B ni experimento reproducible hoy. No reemplazar por loss, perplexity o solo Python |

## Protocolos: qué procede del artículo y qué falta fijar

### P1. GLUE: conservar puntuaciones y ambigüedades

`llm_finetuning.py:124`, `calculate_glue`, utiliza `evaluate.load("glue", task)`.
Con varias salidas añade `combined_score` como media aritmética. En
`Trainer.train_epoch:214–219` el CSV guarda ese combinado o el primer campo
si solo hay una métrica. El artículo rotula todo como accuracy y no explica
por tarea esta agregación. Este mapeo describe el **código**, no una
definición adicional atribuida al paper:

| Tarea | Campos a conservar del evaluador | Escalar seleccionado por upstream | Unidad / sentido |
| --- | --- | --- | --- |
| MNLI matched, SST2, QNLI | `accuracy` | accuracy | Fracción [0,1], mayor mejor |
| MRPC, QQP | `accuracy`, `f1`, `combined_score` | (accuracy + F1)/2 | Fracción [0,1], mayor mejor |
| CoLA | `matthews_correlation` | Matthews | Coeficiente [-1,1], mayor mejor |
| STS-B | `pearson`, `spearmanr`, `combined_score` | (Pearson + Spearman)/2 | Coeficiente [-1,1], mayor mejor |

El evaluador y su revisión deben fijarse antes de ejecutar. Conservar los
componentes explica el escalar existente; no son métricas nuevas. No se
multiplicará por100 al comparar con Tabla3. No se publicará `Avg` de ocho
columnas hasta aclarar SST2 duplicado. Un promedio propio de siete tareas
tendrá nombre/composición explícitos, separado del promedio publicado.

El código reserva los primeros16 ejemplos de train para calibración;
evalúa `validation_matched` en MNLI y `validation` en las otras tareas
(`llm_finetuning.py:750–871`). El script fija longitud128, batch16, bf16,
momentos float32, evaluación cada100 iteraciones, d512,r16, LR5e-5,
CheckFreq1000/inicio de época, α=.3, `--timeout 1`. El paper confirma batch,
LR, frecuencia, umbral y hora; no todos los detalles del script. El timeout
usa `train_start_time`, mientras el campo CSV `time` descuenta evaluaciones.
Guardar límites efectivos y reloj externo; fijar cómo se evalúa el
checkpoint al presupuesto sin inventar que los autores usaron “último”
o “mejor”. D09 exige publicar parámetros realmente entrenados.

### P2. Alpaca: misma definición de perplexity

La ruta usada es `calculate_ppl` (`llm_finetuning.py:108–122`), no
`calculate_perplexity`. Calcula `exp(mean(loss_por_lote))`, sin ponderar
explícitamente por número de tokens de cada lote. Cuando faltan labels,
usa `input_ids`; no introduce aquí máscara -100 para prompt/padding.
El formato instruction/input/output está en `data.py:25–27`; el split
es 90% train por orden, siguientes16 calibración, resto evaluación. La
tokenización, truncado, padding, batch de evaluación, `eval_prob`, lotes
omitidos y semilla pueden alterar el resultado y deben registrarse.

No sustituir silenciosamente esa ppl por exp(NLL total/tokens válidos),
por respuesta solamente o por otra ponderación. Una ppl alternativa,
si se solicita, irá a diagnósticos con su propia definición. El paper
no detalla estos aspectos: no se afirma equivalencia exacta entre la
definición del código y los números publicados solo por su nombre.

Antes de E03 falta resolver D06-D08/D12/D15-D16: Alpaca citado frente a
Alpaca GPT-4 del repo; α=.5 paper frente a .8 CLI; Zero LR1e-5 frente a
1e-4 script; LoRA rango16 figura frente a32 script; flags del scheduler;
comparador de checkpoint que favorece mayor ppl. Se conservarán perfiles
paper/script distintos. No se aplicó ninguna resolución en esta revisión.

### P3. Tiempos, throughput y solapamiento

Conservar magnitudes originales y denominadores explícitos:

- E04: `throughput = iteraciones_completadas / segundos_medidos`;
  `normalized_throughput = throughput_método / throughput_Zero`.
- E05: `slowdown = tiempo_Zero / tiempo_cómputo_GPU_referencia`.
  Documentar FWD, BWD y UPD incluidos; no intercambiar con el denominador E04.
- E06: latencia real en s/iteración, antes de cualquier escalado de tokens.
- E03: h transcurridas realmente. La reducción a igual calidad necesita
  umbral de ppl fijado y alcanzado por ambas rutas; si no se alcanza,
  registrar “no alcanzado”, no extrapolar. Una interpolación se declarará
  como derivación y conservará los dos puntos medidos.

La instrumentación pendiente debe registrar calentamiento, ventana,
repeticiones, sincronización y finalización del trabajo CPU/streams.
El plan local propone al menos tres repeticiones para rendimiento; no
atribuye al paper ese número. Tiempo completo, tiempo de entrenamiento,
evaluación, carga, calibración y guardado se registrarán separados para
que las exclusiones sean visibles. Se conserva el reloj upstream con otro
nombre; no se lo renombra wall-clock total. E03 se graficará desde tiempo
medido, no desde constantes de `draw_loss_gpt2_large.py`.

Hay infraestructura parcial en `para_offload_modules.py:739–796`
(warmup, temporizador, NVTX, pico allocated) y `Scheduler.report:504`.
No existe aún un escritor aceptado de E04-E06 ni un desglose del camino
crítico. `Trainer.offload_profile:333` imprime datos; su bloque que escribe
`profile.csv` está después del `return` de la línea418, por lo que no cuenta como salida
implementada en esa ruta. Además, su estimación de UPD crea Adam y hace
`zero_grad()` antes de `step()` sin asignar gradientes en ese bucle
(líneas390–401): no usarla sin validar como coste real de UPD.
Tiempos de actividades solapadas no se sumarán como segmentos exclusivos
de Fig.2/7a; faltan trazas y regla de atribución.

### P4. Error de estimación

En `simluation/lspoffload/utils/benchmark_compressor.py`,
`BenchmarkContext` prepara por defecto128 muestras de calibración y128 de
test, semilla42, longitud1024 y batch8. Acumula gradientes, aplica clipping
1.0 y guarda ambas colecciones en CPU. Son defaults del código, no valores
confirmados por Fig.9. Su memoria debe medirse antes del barrido.

`simluation/lspoffload/optim/dummy.py:step` calcula por matriz
`norm(G - decompress(compress(G))) / norm(G)` y devuelve la media no
ponderada entre matrices. Guardar numerador, denominador y agregación;
un gradiente de norma cero se declarará indefinido, sin convertirlo en
error0. El paper no especifica la agregación entre matrices del gráfico.
La comparación SVD usará la ruta correspondiente; no suponer que el
compresor principal implementa el mismo GaLore. Faltan un lanzador de
Fig.9, módulos exactos, revisiones y verificación de la segunda ruta.
Un estudio con tensores sintéticos es diagnóstico.

### P5. DeepSeek: calidad medida y tiempo simulado

El apéndice p.12 declara tres pasos: entrenar con memoria GPU suficiente,
medir tiempo de offload por paso con batch viable y combinar/escalar por
batch de tokens. **120h/15h/30h son presupuestos equivalentes simulados**,
no duración continua medida de offloading. También afecta a la selección
de checkpoints de Tabla4, aunque su caption no lo repita.

`simluation/example/run_sft_lsp.sh` fija batch16×acumulación8, longitud1024,
cinco épocas; `sft_main_lsp.py:410–459` registra en TensorBoard una pérdida
agregada sobre el intervalo de logging, con acumulación, redondeada a cuatro
decimales. No llamarla pérdida bruta por microbatch. Baselines usan otros
entrenadores. Faltan exportador común y verificación de agregación, pasos,
tokens y objetivo. `result_ref/code_train_losses_1.3B.csv` es referencia
de autores, no recolector local.

Una reconstrucción futura conservaría `source_quality_run`,
`source_profile_run`, pasos, tokens, factor de escalado y
`time_kind=simulated`, separada de `time_kind=measured`.
Offloading real local sería otra configuración registrada, sin heredar
los tiempos de esa simulación.

### P6. HumanEval: información que falta

Tabla4 enumera python, java, cpp, js, ts, php, `Avg`, `GPU Mem` y `Time`.
El paper habla de evaluation accuracy y cita HumanEval; eso **no fija**
pass@1/pass@k ni el procedimiento multilenguaje. La revisión del árbol
de99 archivos no localizó un harness que produzca esas puntuaciones.

Antes de ejecutar hacen falta: dataset/revisión por lenguaje, harness/commit,
número/IDs de problemas, prompt y extracción de código, plantilla de chat
si aplica, decodificación/semillas, muestras por problema, definición de
métrica/k/estimador, compiladores/runtimes, timeouts y errores, denominador
y promedio, checkpoint seleccionado. No rellenar campos con defaults de
otra biblioteca ni inventar equivalencia con un benchmark parecido.
Se consultará esa información al pasar a E09; esta matriz registra el
bloqueo sin ejecutar el evaluador.

La columna de memoria dice GB, pero no define pico allocated, reserved,
memoria de dispositivo ni fase. No se elegirá una para afirmar igualdad
con Tabla4. La telemetría local conservará bytes y tipo explícitos,
separada hasta aclarar la definición.

## Cobertura fuera del conjunto inicial

| Resultado del paper | Métrica y baseline que se conservarían | Situación con este hardware / información | Código y destino si se habilita |
| --- | --- | --- | --- |
| Fig.5b p.7; barras Llama-3B Fig.2 p.3 y Fig.6 p.7 | Perplexity frente a tiempo (h), slowdown y throughput normalizado; Zero/LoRA y variantes LSP d256/512/1024,r16 según figura | **Diferido, viabilidad no demostrada**. Solo pesos3B a16bits ≈5.59GiB; margen reducido de VRAM, calibración/baseline muy exigentes en RAM7.61GiB. Batch16 original no validado. No afirmar OOM medido. Reducir batch sería adaptación | `scripts/run-llama-3b.sh`; salidas equivalentes E03-E05, en run propio |
| Fig.5d, Fig.7a derecha, Tabla4 inferior; pp.6–7/12, DeepSeek-6.7B | Train Loss vs h simuladas; latencia s/iter; score multilenguaje, GB. Zero frente a LSP d2048,r8; presupuestos equivalentes15h/30h | **No viable en ruta original**. Pesos16bits ≈12.48GiB >7.96GiB total, antes de activaciones. La calidad en GPU suficiente tampoco cabe. Evaluación de tabla además bloqueada por P6 | Scripts6.7B existentes; no recolector local habilitado ni archivo de medición generado |
| Tabla1 p.3 y barras Llama-7B Fig.2/6 | Memoria GB; ancho de banda GB/s; FWD/BWD/UPD CPU s/capa, GPU ms/capa; Zero y referencia GPU | **Modelo completo no viable**. Pesos16bits ≈13.04GiB >7.96GiB. Un microbenchmark de una capa podría caber, pero sería parcial y no validaría totales del modelo | `llm_finetuning.py:572`, `layer_profile`, es parcial: usa `torch.optim.Adam`, no el fused Adam de Tabla1. Futuro microbenchmark: `R/diagnostics/layer_profile.csv`, declarado parcial |
| Tabla5 p.12 y barras “GPT2-1.3B” Fig.2/6 | Magnitudes por capa/GB/GB/s; slowdown/throughput según figura | **Bloqueado por identidad y memoria condicional**, no imposibilidad probada por VRAM. Rótulo1.3B/40capas no identifica checkpoint; no sustituir por GPT2-XL, OPT ni DeepSeek. Estado de optimizador tabulado7.8GB deja escaso margen en RAM Linux total, sin demás tensores | `layer_profile` es parcial, no protocolo validado. Tras identificar modelo: `R/paper/layer_profile.csv`, `memory_components.csv`, `transfer_bandwidth.csv`; escritores pendientes |
| Tabla2 p.4; Fig.1 p.1, Fig.3 p.4, Fig.4 p.5 | Tabla2: expresiones teóricas de memoria/rango; figuras: arquitectura, planificación y proyección | **No son benchmarks adicionales**. No convertir fórmulas/dibujos en mediciones. Pueden motivar diagnósticos de E04/E07 | Sin resultado experimental por copiar fórmulas; trazas/contabilidad propias en diagnósticos |

Los equipos originales (A1000/32GB RAM y RTX4090/252GB RAM) no están
disponibles. Las repeticiones viables en RTX5050 serán reproducciones
parciales en otro hardware, sin expectativa de igualar valores absolutos.
GaLore1.3B aparece con7.9GB en Tabla4, cerca de la VRAM total local y por
encima de la libre observada: se mantiene **condicional**, sin declarar
OOM ni garantizar que quepa por tener “8GB”.

## Diagnósticos adicionales, separados de resultados del paper

| Diagnóstico | Relación con paper | Métrica / unidad y referencia | Estado de implementación y archivo |
| --- | --- | --- | --- |
| Entorno, kernel y copia fijada | Prerrequisitos, sin tabla/figura reproducida | Versiones y bytes libres; error absoluto de delta/M/V frente a fórmula de cada ruta; igualdad tras copia1MiB por dirección | **Implementado y ejecutado**: `tools/validate_environment.py`, `results/environment-validation-locked/diagnostics.json`; resumen versionado `configs/environment_validation.json`. No mide ancho de banda ni entrenamiento |
| Colisiones, signos y reinicio de momentos | Definición1 p.4, Alg.1 p.5; no resultado de Fig.9 | Conteo de colisiones, valores iniciales, sumas absolutas M/V antes/después; tensores sintéticos d8 | **Implementado y ejecutado** en mismo JSON. Demuestra N15, no lo corrige; no compara convergencia |
| Concurrencia serial/FCFS/LCFS y reanudación | Alg.3 p.11, requisitos locales de fidelidad | Error absoluto/relativo con mismos pesos/datos/proyectores; actualizaciones pendientes/perdidas; estado del paso siguiente | **Falta**: `R/diagnostics/scheduler_correctness.json`, `resume_check.json`, `R/trace/`. No sustituir por kernel aislado |
| Diagnóstico corto MRPC/SST2 o GPT2 | E01/E03 como candidatos posteriores | Loss finita, pasos, ejemplos; calibraciones efectivas y persistencia de M/V | **Falta**: `R/diagnostics/smoke.json`, `compressor_events.csv`. No equivale a1h GLUE ni a curva Alpaca completa |
| Telemetría y coste completo | Control local; no equivalencia automática con `GPU Mem` | Bytes allocated/reserved/dispositivo; RSS/cgroup/MemAvailable/swap; fases en s, transferencias bytes y GB/s si cronometradas | **Parcial**: instantáneas y prueba de copia disponibles; **falta monitor continuo** `R/diagnostics/resources.csv`, `timeline.csv`, `transfers.csv` |
| Throughput adicional | Complementa E04, no reemplaza iter/s | Muestras/s, tokens válidos/s, tokens con padding/s; microbatches y actualizaciones separados | **Falta**: `R/diagnostics/throughput_tokens.csv`. Comparar solo con igual regla de conteo |
| Metadatos, fallos y resumen | Reproducibilidad transversal | Configuración, versiones, success/OOM/failed/interrupted, exit code, IDs de fuentes | **Falta supervisor de experimentos**: `R/config.json`, `environment.json`, `summary.json`, logs incrementales. Fallos no se rellenan con ceros |

## Criterio para preparar experimentos

Primero cerrar decisiones y pruebas de algoritmo pendientes, y después
implementar recolectores de las filas elegidas. E01/E02 y LSP/LoRA de E03
son candidatos iniciales; E04/E05 requieren demostrar también viabilidad
del baseline. E06-E08 siguen condicionados. E09 necesita información antes
de definir un protocolo fiel. Los casos6.7B/7B no se prepararán en la ruta
original de esta GPU.

Cada futura fila conservará `experiment_id`, `run_id`, `paper_locator`,
nombre original de métrica, definición efectiva, unidad, baseline_run_id,
`time_kind`, estado y ruta de evidencia. Diferencias de configuración y
métricas adicionales irán identificadas; no habrá resultados de entrenamiento
aceptados mientras `algorithm_accepted` siga siendo false.
