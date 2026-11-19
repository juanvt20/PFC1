# Contraste de las referencias locales y el upstream

Revisión local: 12 de noviembre de 2026 (Lima). Las páginas indicadas son
posiciones del PDF, contando desde 1; no son índices desde cero. Upstream:
`d9dc92a32b7e4829116ba6abec17c280dfb548a4`. Las rutas de código de las tablas
son relativas a `original/upstream/`. Esta revisión inicial no aplicó parches.
Actualización posterior: el usuario autorizó N15, implementado exclusivamente
en `adaptado/` y verificado con tensores completos. N14 se confirmó en
ejecución y detuvo el piloto; véase `VALIDACION_LOCAL.md` y
`configs/adaptation_validation.json`. Las filas siguientes conservan el
contraste con el código original, que permanece intacto.

## Fuentes y cobertura

Se leyó primero `referencias/LSP-OFFLOAD.pdf` (13 páginas), después
`referencias/PFC (1).pdf` (9 páginas). `referencias/STRONGHOLD.pdf`
(15 páginas, incluida la portada editorial) se consultó como contexto:
ventanas, concurrencia, modelo analítico y plataformas originales.
La extracción por página y los PNG quedan locales en `tools/pdf_output/`,
excluidos de Git igual que los PDF. Los hashes están en VALIDACION_LOCAL.md.

Se inspeccionaron visualmente las páginas 1, 3–7 y 10–12 de LSP: Figuras 1–9,
Tablas 1–5, ecuaciones y Algoritmos 1–3. Se revisaron las páginas 5–8 del
PFC: Tabla I girada, ecuación (1), Tablas II/III y Figuras 1–3.
En STRONGHOLD se inspeccionaron páginas 5, 6 y 9: secciones III-B/C/D,
Figura 3, restricciones P1/P2 y sección V-A/B/C. No se realizó una auditoría
matemática exhaustiva del artículo de STRONGHOLD ni de su implementación.

Los tres PDF tienen texto extraíble en todas sus páginas. La extracción
mezcla columnas y contiene duplicados en gráficos; especialmente la Tabla I
girada del PFC produce texto desordenado. Sus datos relevantes se leyeron
en el render, no se reconstruyeron de esa extracción. Las fórmulas se
contrastaron visualmente. No se declara verificado un valor exacto por
digitalización de curvas ni una demostración formal de convergencia.

## Diferencias entre método, código y experimentos

| ID | PDF: página y sección | Código: archivo / función | Hallazgo y consecuencia |
| --- | --- | --- | --- |
| D01 / N02 | LSP p.4, “Efficient and High-quality Offloading via Learned Sparse Projectors”, Definición 1; p.5 “Learned sparse projectors” | `gradient_compressor.py:265`, `CountSketch.__init__`, `get_sketch`; segunda ruta `simluation/lspoffload/compressor/count_sketch.py:68`, `_rand_indices` | La definición exige r posiciones no nulas por fila de P/Q. La ruta principal almacena sus transpuestas y muestrea con reemplazo; pueden repetirse índices de una columna y `scatter_` sobrescribe, no suma. La segunda ruta usa `multinomial(..., replacement=False)`. No son rutas intercambiables. El diagnóstico d=8,r=4 encontró colisiones en 81 de 128 columnas; es un diagnóstico sintético, no una tasa para d=512. |
| D02 / N03 | LSP p.5, “Learned sparse projectors”: inicialización gaussiana | `gradient_compressor.py:269`, `CountSketch.__init__`; `llm_finetuning.py:691` | La opción binaria de la ruta principal usa `randint(0,1)`: todos sus valores iniciales son negativos. La CLI principal usa gaussiana por defecto, coherente con el texto del paper. No atribuir el defecto binario a todas las ejecuciones; la segunda ruta sí usa `randint(0,2)`. El diagnóstico r=4 confirmó valor único -0.5. |
| D03 / N04 | LSP p.5, Algoritmo 1 línea 6 y ecuación (3) | `gradient_compressor.py:312`, `LearnedCountSketchCompressor.train`, `_fit` | El paper usa error absoluto más regularización; el código minimiza `norm(approx-t)/norm(t)` sin término beta explícito, con un número fijo de iteraciones. El paper indica parar por umbral o timeout. No añadir beta ni sustituir la función objetivo sin decisión explícita. |
| D04 / N04 | LSP p.5, “Learned sparse projectors” y Algoritmo 1 líneas 5–6 | `gradient_compressor.py:372`, `LearnedCountSketchCompressor.fit` | El primer fit llama a `_fit(..., [], n_iter=0)` tanto común como específico. Inicializa, pero no aprende de gradientes. El aprendizaje efectivo debe comprobarse en un fit posterior; una prueba que termina antes no valida proyectores aprendidos. |
| D05 / N15 **nuevo** | LSP p.5, Algoritmo 1 líneas 7–9: transporte de M/V | `gradient_compressor.py:433–438`, `LearnedCountSketchCompressor.fit`; `para_offload_modules.py:248`, `ParallelOffloadAdamMMOptim.init`; `:557`, `Adam.init` | Tras calcular M/V transportados, `fit` llama incondicionalmente a `module.init`, que invoca `optim.init` y crea ceros. Se reprodujo con dos fits, matriz 16×16, d=8,r=2, float32. La evaluación previa identificó las fórmulas de transporte pero omitió su posterior reinicio. Es un bloqueo científico: no se ha corregido. La ruta SVD también llama `module.init` tras transformar; requiere revisión propia si se usa. |
| D06 / N05 | LSP p.11, “Instruction Fine-tuning on Alpaca”: α=0.5; “GLUE Experiment”: α=0.3 | `llm_finetuning.py:690`; `scripts/run-gpt2-large.sh`, `run-llama-3b.sh`, `run_glue.sh` | Los lanzadores Alpaca heredan 0.8; GLUE sí fija 0.3. CheckFreq=1000 y comienzo de época se implementan con el índice local reiniciado en `Trainer.train_epoch`. Separar perfiles paper/script y declarar cualquier cambio. |
| D07 / N06 | LSP p.6, “Layer-wise Schedule for Maximal Parallelism”; p.11, Algoritmo 3 y “Avoiding blocking”; p.7 Fig.6 | `para_offload_modules.py`, `Scheduler`, `ParallelOffloadAdamMMOptim.forward/backward`; `llm_finetuning.py:663–669,1000–1006`; scripts GPT2/Llama | Los lanzadores Alpaca no fijan transición/delays; heredan ceros. La línea base rotulada Zero no pasa `--scheduler-zero`. No basta ejecutar el script para afirmar que reproduce Zero estándar o la ablación FCFS/LCFS. GLUE sí fija parámetros, pero usa optimizador en GPU. El orden antes de forward y los hooks pendientes requieren validación por trazas. |
| D08 / N07 | LSP p.11, “Instruction Fine-tuning on Alpaca”; p.7 Fig.5 leyendas | `scripts/run-gpt2-large.sh`, `run-llama-3b.sh`; `scripts/draw_loss_gpt2_large.py` | Paper: LR Zero=1e-5 y LoRA rango16 en figura. Script GPT2: Zero=1e-4, LoRA rango32; script Llama: rango8. La gráfica GPT2 rotula rango16 aunque lee la salida del script rango32. No escoger silenciosamente la mejor configuración de cada fuente. |
| D09 / N08 | LSP p.6, “Accuracy validation ... GLUE”, Tabla3; p.11 “GLUE Experiment” | `llm_finetuning.py:898–901,923–938`, congelación y sustitución de módulos; `para_offload_modules.py`, `replace_linear_layer_with_parallel_linear` | El nombre FullParam/Zero no demuestra actualización de todos los parámetros del modelo: cuando hay offload se actualizan matrices seleccionadas y el clasificador, con otros parámetros congelados. Publicar la lista y conteo efectivos. |
| D10 / N11 | LSP p.11, “Experiment Configurations and Further Results”: semilla fija en código | `llm_finetuning.py:670,712–713`; lanzadores principales | `--seed` es None por defecto y los scripts no la fijan. No se conoce una semilla efectiva original por esos lanzadores. Registrar una semilla local no la convierte en semilla de los autores. |
| D11 | LSP p.6, Tabla3 y p.11 Fig.8 / “GLUE Experiment” | `scripts/run_glue.sh`; `llm_finetuning.py`, preparación de GLUE | Tabla3 repite SST2; Fig.8 muestra siete tareas. El bucle activo del script solo ejecuta MNLI y usa `--offload cuda:0`. No reconstruir una octava tarea ni llamar validación CPU-GPU a esa ruta. El promedio publicado es el de su tabla, no el de nuestro subconjunto diagnóstico. |
| D12 / N12 | LSP p.6, “End-to-end evaluation”; p.11 “Instruction Fine-tuning on Alpaca” | `data.py:23`, `InstructionTuningDataset`; `llm_finetuning.py:772–793` | El paper denomina el dataset Alpaca y cita Stanford; el código lee `datasets/alpaca_gpt4_data.json` (la procedencia propuesta está en README upstream). No asumir mismos ejemplos. El split en código es por orden: 90% train, luego calibración, luego evaluación. La clave de caché omite longitud/revisión y otras decisiones. Fijar archivo/hash, tokenizer y caché por perfil. |
| D13 | LSP p.2 “Other methods ...” y p.3 Fig.2: fp16; p.6 Tabla4: bfloat16 | Scripts GLUE/Alpaca: `--dtype bfloat16`, `--optim-dtype float32` | No hay una única precisión explícita coherente para todos los pasajes del paper. Conservar la distinción por experimento; no convertir bf16 a fp16 como cambio neutro. |
| D14 | LSP p.7 Fig.5c/d; p.12 “Instruction Fine-tuning with DeepSeek-Coder” | `simluation/example/finetune_deepseekcoder.py`, `llm_finetuning.py`, scripts de gráficas | Las curvas DeepSeek combinan entrenamiento con memoria GPU suficiente y perfil offload escalado. Las 120h/15h/30h son presupuestos simulados equivalentes, no evidencia de entrenamiento continuo local con offloading. No usar `simluation/` como validación automática del sistema completo. |
| D15 | LSP p.7 Fig.5a/b y caption de suavizado/desviación | `scripts/draw_loss_gpt2_large.py:1–36`, equivalente Llama; `llm_finetuning.py:225`, `Trainer.train_epoch` | El código gráfico reconstruye horas con constantes por iteración, sustituye el primer ppl por 2.5 y escala la desviación móvil por 0.15. El entrenador excluye tiempo de evaluación de su reloj interno. Corrección de la evaluación previa: tampoco se debe suponer wall-clock bruto en Alpaca por no estar rotulado “simulado” en Fig.5a/b. No demuestra que las cifras publicadas sean falsas; limita lo que reproducen los scripts disponibles. |
| D16 / N09 | LSP p.7, “Comparison with Zero-Offload”, Fig.5a/b: menor perplexity es mejor | `llm_finetuning.py:204,222`, `Trainer.train_epoch` | La selección de checkpoint para Alpaca usa `score > best_score`, favoreciendo mayor perplexity. Los CSV brutos y la selección de checkpoint son evidencias diferentes. No cambiar el comparador sin registrar su efecto sobre evaluación. |
| D17 / N10 | LSP p.5 Alg.1 mantiene P/Q/M/V; p.12 resultados por checkpoint | `llm_finetuning.py:222,297–313`, guardado y `load_ckpt` | Se guarda estado del modelo y CSV; no todos los momentos/proyectores/RNG/cursor/scheduler. El paper no promete aquí un formato exacto de reanudación: es una carencia de reproducibilidad para nuestro protocolo, no una contradicción textual demostrada. |
| D18 / N01 | LSP p.11, “Implementation”: Adam fusionado, sin fórmula de epsilon | `kernels/fusedAdam_cpu.cpp`, `fused_adam_cpu_impl`; `fusedAdam_cuda.cu`, `fused_adam_cuda_impl`; `para_offload_modules.py`, `Adam` | Diferencia interna de código: CPU divide por sqrt(v_hat+eps); GPU por sqrt(v_hat)+eps. Adam local usa eps=1e-4. El paper no permite escoger cuál corregir. Ambas rutas se probaron contra su propia fórmula float32; pasar esas pruebas no establece equivalencia CPU/GPU. |
| D19 / N13–14 | LSP p.11 Alg.3 línea5; p.6 planificación sin pesos obsoletos | `kernels/fusedAdam_cuda.cu`, lanzamiento sin stream explícito; `para_offload_modules.py`, `Scheduler.clear`; `llm_finetuning.py`, evaluación / fin de época | Son riesgos pendientes de ejecución concurrente, no fallos demostrados solo por lectura. La prueba de kernel en stream por defecto y la transferencia aislada no cierran este criterio. |

## Precisiones de evaluación añadidas el 13 de noviembre de 2026

Estas precisiones sustentan [MATRIZ_EXPERIMENTOS.md](MATRIZ_EXPERIMENTOS.md).
No se modificaron métricas ni implementaciones originales.

| ID | PDF: página y sección | Código: archivo / función | Diferencia o información ausente |
| --- | --- | --- | --- |
| D20 | p.6 Tabla3; p.11 Fig.8 / “GLUE Experiment” | `llm_finetuning.py:124`, `calculate_glue`; `Trainer.train_epoch:214–224` | El paper rotula accuracy/Eval Acc sin precisar agregación por tarea. El código añade media de campos para MRPC/QQP y STS-B y guarda solo el escalar en columna `glue`; los componentes quedan en stdout. Registrar todos los campos y el combinado; no convertir MCC/correlación en accuracy ni reconstruir el Avg de Tabla3. |
| D21 | p.7 Fig.5a/b; p.11 “Instruction Fine-tuning on Alpaca” | `llm_finetuning.py:108`, `calculate_ppl`; `data.py:InstructionTuningDataset.format`; collator/split en `llm_finetuning.py:772–793` | La ruta efectiva usa exp(media de pérdidas por lote), sin ponderación explícita por tokens; labels=input_ids sin enmascarado explícito de prompt/padding en esta función. El paper no fija esa definición. No reemplazarla silenciosamente por ppl ponderada por tokens o de respuestas solamente. |
| D22 | p.7 Fig.7a; p.12 “Instruction Fine-tuning with DeepSeek-Coder”, perfil1.3B de1×384 tokens | `README.md`, comandos de perfil Fig.5c; `para_offload_modules.py:653` | El comando LSP fija seq_len384; el Zero lo omite y hereda1024. Ejecutar literalmente ambos comandos no compara igual carga. Registrar la divergencia y fijar perfil común antes de evaluar latencias. |
| D23 | p.3 Tabla1/Fig.2; p.7 Fig.6/7a | `llm_finetuning.py:333`, `Trainer.offload_profile`; `:572`, `layer_profile` | `profile.csv` está después de un return en la primera función. La estimación GPU de UPD llama zero_grad/step sin asignar gradientes en ese bucle; no acredita un UPD real. `layer_profile` usa torch.optim.Adam en vez del kernel fusionado descrito para Tabla1. Son limitaciones del recolector disponible, no prueba de cómo se obtuvieron los números publicados. |
| D24 | p.6 Tabla4; p.12 “Instruction Fine-tuning with DeepSeek-Coder” | README y árbol completo auditado, sin harness HumanEval localizado | No están fijados protocolo multilenguaje, métrica/k, generación, revisión y agregación. Tampoco se define qué medida concreta es GPU Mem. Los presupuestos de checkpoints proceden de simulación. No completar esas ausencias con supuestos ni presentar una loss como puntuación HumanEval. |

## Ambigüedades del propio artículo

En p.5, ecuación (3), la regularización es β(||P||F+||Q||F).
En Algoritmo 1 línea6 aparece β(||P||F²+||Q||F). No escoger una
interpretación silenciosamente. En la misma página el texto de ecuación
(2) atribuye a ambos proyectores dimensiones n×d aunque Definición 1 p.4
usa m×d y n×d; para matrices rectangulares se sigue la definición
dimensional al explicar, sin afirmar una corrección de implementación.

El código devuelve deltas ya negativos y escalados por LR y los suma al
peso (`Adam.step`, `ParallelOffloadAdamMMOptim.backward`, `_update`).
El pseudocódigo resta η por separado. Es una diferencia de convención
de interfaz, no evidencia de un signo erróneo ni razón para aplicar LR dos veces.

El análisis de convergencia de pp.5–6 y la prueba de p.10 consideran
actualizaciones de gradiente con CheckFreq=1 y supuestos de error acotado.
No es una demostración de equivalencia para Adam fusionado, recalibración
cada 1000 pasos, colisiones, bf16 o hooks concurrentes. L-smoothness no
implica por sí sola convexidad, pese a la frase en p.5. No se ha realizado
una revisión formal independiente de la demostración.

## Correcciones para el PFC y la evaluación anterior

| Fuente | Contraste y acción documental |
| --- | --- |
| PFC p.5, Tabla I fila LSP-Offload | Atribuye OPT-1.3B / OPT-6.7B. LSP p.6 “End-to-end evaluation” nombra GPT2-774M, Llama-3B y DeepSeek-Coder-1.3B/6.7B; p.12 Tabla5 usa el rótulo “GPT2-1.3B”, no OPT. Proponer sustituir la atribución en la futura fuente editable del PFC; no se modificó el PDF. No equiparar GPT2-1.3B, GPT2-XL y OPT-1.3B por el número nominal. |
| PFC p.3 §II-B y p.5 Tabla I | La descripción de capacidad 1.3B/4GB y 6.7B/24GB debe incluir el protocolo simulado de DeepSeek de LSP p.12. La reducción 33.1–62.5% se presenta para Alpaca p.7; no extrapolarla como aceleración local demostrada o resultado continuo DeepSeek. |
| PFC p.6 §III-B, ecuación(1) | La compresión G=Pᵀ∇WQ es dimensionalmente consistente con LSP p.5 Alg.1. Conviene completar la descripción con aprendizaje, recalibración, transporte de momentos y validación FCFS/LCFS; la implementación publicada no satisface automáticamente todos esos puntos. |
| PFC pp.7–8 §IV | STRONGHOLD mide GPT 51/124/209M FP32, WikiText-103, secuencia256 y batch2. LSP prioriza RoBERTa/GLUE y GPT2-Large/Alpaca bf16. No comparar directamente throughput/calidad como si fueran la misma tarea. Los límites sobre una ejecución, calentamiento y ausencia de convergencia del PFC se conservan. |
| STRONGHOLD PDF pp.5–7 §III-C/D/E, Fig.3; p.9 §V-A/B | Confirma ventana de estados y actualización CPU concurrente, frente a pesos residentes y estados comprimidos de LSP. Plataforma V100 32GB, 755GB RAM, PyTorch1.10.2/CUDA11.2. No copiar sus dependencias antiguas a LSP ni modificar su código. |
| Etapa A y README locales | Ya existe acceso local a disco, Git, Docker y GPU, y extensión compilada con fuentes originales. Esto actualiza las limitaciones históricas de acceso; no equivale a entrenamiento ni a corrección del algoritmo. |

La evaluación de viabilidad se mantiene **condicional**, con D05/N15 como
nuevo bloqueo prioritario. La preparación de entorno puede avanzar; el
fine-tuning de calidad y la comparación con baselines esperan una decisión
explícita sobre cambios matemáticos y sus pruebas. No se descargaron modelos
ni datasets y no se ejecutó entrenamiento largo.

## Propuesta N15 de la revisión inicial y estado posterior

Se propone una corrección aislada en una futura copia `adaptado/upstream/`:
separar la creación inicial de estados del optimizador de la actualización
de buffers al recalibrar, para conservar los M/V ya transportados y el
contador de pasos. Debe probarse en CPU y GPU, con dos recalibraciones y
comparación explícita contra las fórmulas del Algoritmo1. La prueba actual
observa el fallo, pero no valida todavía ese parche propuesto.

Esta propuesta cambia la trayectoria de entrenamiento respecto al código
publicado; requiere autorización antes de implementarse. No incluye elegir
epsilon, añadir regularización beta, cambiar muestreo o sustituir Adam.
Esas diferencias mantienen decisiones separadas y no se corregirán de forma
implícita bajo una etiqueta de compatibilidad. La copia original se mantiene
como referencia byte por byte.

El usuario autorizó después esta propuesta y el parche0001 ya la implementa
para LearnedCountSketch. La inicialización y dos recalibraciones pasan
comparaciones completas M/V en CPU/GPU; las otras diferencias no se
cambiaron. La corrección de `Scheduler.clear`, ahora confirmada como N14,
se propuso por separado y sigue pendiente de autorización. No se ejecutó
piloto ni comparación de rendimiento debido a ese bloqueo.
