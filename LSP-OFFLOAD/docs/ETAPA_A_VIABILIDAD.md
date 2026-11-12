# Etapa A: viabilidad técnica de LSP-Offload

Fecha: 12 de noviembre de 2026. Proyecto: «Evaluación y Adaptación de Estrategias de Paralelización para el Entrenamiento de Modelos de Lenguaje en Entornos de Recursos Limitados».

## Dictamen y alcance de la evidencia

**Es razonable comenzar una reproducción parcial y fiel con el hardware actual. No está demostrada todavía su ejecución en la RTX 5050.** Los objetivos iniciales son RoBERTa-base/GLUE para calidad y GPT-2 Large/Alpaca para la ruta completa CPU-GPU. No hace falta asumir una ampliación de RAM o SSD para comenzar. El pico de calibración, la memoria fijada de WSL2 y la corrección de la extensión son las principales condiciones de avance.

Se revisó el PDF adjunto de LSP-Offload de 13 páginas, incluido el apéndice; el PFC adjunto para situar la metodología; el árbol completo del upstream y sus fuentes Python, C++/CUDA y shell; y la estructura de PFC1. Se inspeccionaron los mecanismos de compresión, recalibración, actualización de momentos, autograd, planificación, datos, evaluación, guardado y gráficas. La revisión estática no demuestra la corrección ni el rendimiento en ejecución.

Se comprobó mediante GitHub que `juanvt20/PFC1` es **público**, con permisos de lectura y escritura en esta conexión, y que `main` partía del commit `0b70b6a8a8572fa6765c2a4a616520763e6e922d`. Su raíz contenía `.gitignore` y `STRONHOLD/`; no había LSP-Offload ni un `AGENTS.md` en el árbol inspeccionado. Se preserva literalmente el nombre `STRONHOLD` existente.

Upstream auditado: `gulang2019/LSP-Offload`, commit `d9dc92a32b7e4829116ba6abec17c280dfb548a4`, de 20 de agosto de 2024; árbol `789020101c7d4fc5537cd0822656b7026f8db932`. El código precede a la publicación AAAI 2025; por ello no se supone que cada script coincide exactamente con el artículo. El manifiesto conserva rutas, tamaños y hashes Git de todos sus archivos.

Se recuperaron 44 archivos de fuente/configuración, incluido el módulo vacío `simluation/lspoffload/__init__.py`, y se verificaron sus bytes frente a los hashes de blobs GitHub. Se analizaron sintácticamente 30 Python y 10 shell, sin errores de sintaxis. No se compilaron extensiones ni se ejecutó entrenamiento. No se recuperaron los binarios precompilados ni se reutilizaron las mediciones de `result_ref/` como datos propios.

## 1. Qué técnica hay que conservar

Para pesos $W\in\mathbb{R}^{m\times n}$, LSP mantiene proyectores $P\in\mathbb{R}^{m\times d}$ y $Q\in\mathbb{R}^{n\times d}$ dispersos, con $r$ posiciones no nulas por fila en la definición del artículo. El gradiente comprimido es $G=P^T\nabla WQ$. La CPU calcula la actualización y los momentos del optimizador en el espacio $d\times d$; la GPU recibe el delta y lo reconstruye como $P\Delta SQ^T$ para actualizar los pesos. No es LoRA ni una compresión aleatoria fija.

Los valores de los proyectores se aprenden con gradientes de calibración, minimizando el error relativo de reconstrucción. Periódicamente se evalúa ese error y se decide recalibrar según un umbral. Cuando cambia el subespacio también se trasladan los momentos primero y segundo al nuevo espacio. La implementación concreta tiene diferencias respecto al pseudocódigo, registradas en COMPATIBILIDAD.md; no se añade una fórmula distinta para ocultarlas.

El sistema conserva forward/backward en GPU, momentos y actualización en CPU, buffers fijados en RAM, streams CUDA para transferencias en ambas direcciones y hooks por módulo para el orden FCFS/LCFS. Debe esperarse la actualización que necesita cada forward. No se introducirá una actualización con pesos obsoletos como sustitución del planificador.

En el código principal, `gradient_compressor.py` implementa `LearnedCountSketchCompressor`; `para_offload_modules.py` implementa `ParallelOffloadAdamMMOptim`, `Scheduler`, `Adam` y la sustitución de capas; `kernels/` contiene el Adam fusionado. La carpeta `simluation/` contiene otra ruta de entrenamiento, útil para estudiar calidad, que no demuestra por sí sola el sistema de offloading.

## 2. Experimentos originales y correspondencia con el código

Los números de memoria y presupuestos de esta sección pertenecen a los autores, salvo donde se indique expresamente «estimación». No son mediciones de esta laptop.

| Evaluación | Modelos y datos | Configuración principal publicada | Hardware original / naturaleza | Viabilidad inicial |
| --- | --- | --- | --- | --- |
| Tabla 3 y Fig. 8: calidad | `roberta-base`, denominado 117 M en el paper; GLUE | Batch 16, longitud 128 en script, LR 5e-5, d=512, r=16, CheckFreq=1000 o inicio de época, umbral 0.3, presupuesto 1 h | Validación del método; el script usa optimizador en `cuda:0`, por lo que no prueba offloading CPU | Prioridad 1: muy probable, pendiente de comprobar memoria y extensión |
| Fig. 5a: instrucciones | `gpt2-large`, aproximadamente 774 M; Alpaca en el texto, archivo Alpaca GPT-4 en el código | Batch 4, longitud 512; d=512; r=8,16,32; CheckFreq=1000; LSP LR 1e-4 | A1000 Laptop 4 GB, i7-12800H, **32 GB RAM** | Prioridad 2: candidato fuerte, con condición de RAM en calibración y baseline |
| Fig. 5b | `openlm-research/open_llama_3b`, no Meta Llama 3; Alpaca GPT-4 en código | Batch 16, longitud 512; d=256,512,1024; r=16 | RTX 4090 24 GB, Threadripper 3970X, **252 GB RAM** | Diferir: 3B pesos a 16 bits ocupan unos 6 GB decimales, sin activaciones ni temporales |
| Fig. 5c y Tabla 4 superior | `deepseek-ai/deepseek-coder-1.3b-base`; `theblackcat102/evol-codealpaca-v1`; evaluación HumanEval multilenguaje | Batch efectivo 128, longitud 1024, AdamW, 5 épocas, cosine; d=1280,r=4, LR 1e-4; perfil offload batch 1, longitud 384 | Perfil en laptop; curva temporal **simulada combinando calidad y perfil**; presupuesto equivalente 120 h | Condicional, después de GPT-2; no prometer réplica completa |
| Fig. 5d y Tabla 4 inferior | `deepseek-ai/deepseek-coder-6.7b-base`, mismo dataset de código | Batch efectivo 64, longitud 1024, 1 época, cosine, LR 1e-4 y mínimo 5e-5; d=2048,r=8; perfil batch 4, longitud 1024 | Perfil en workstation; curva temporal simulada; 15/30 h equivalentes | No viable con ruta original en 8 GB |
| Fig. 6: ablación de sistema | GPT2-774M / «GPT2-1.3B» y Llama-3B/7B | Zero, Zero+planificación por capas, LSP con varios d, r=16; referencia de cómputo GPU | GPT en laptop; Llama en workstation | Subconjunto GPT-2 viable sujeto a medición; no todos los lanzadores están publicados |
| Fig. 2 y Tablas 1/5 | Perfil/motivación de Zero, modelos GPT y Llama | FWD, BWD, UPD, comunicación y tiempo no solapado | Dos equipos anteriores | Reproducir el protocolo que pueda identificarse, no sus tiempos absolutos |
| Fig. 7b y Fig. 9 | Error de estimación de proyectores en DeepSeek-Coder-1.3B | Variaciones de d,r; comparación con SVD de GaLore; calibración y validación | Análisis de compresores | Posible subconjunto posterior; pipeline completo de datos/evaluación no está fijado |

El artículo llama «GPT2-1.3B» a un modelo de sus perfiles y reporta 40 capas, pero los scripts de Alpaca utilizan `gpt2-large`; no se debe identificar automáticamente ese rótulo con `gpt2-xl` ni con OPT-1.3B sin evidencia adicional. El PFC adjunto menciona OPT-1.3B en su tabla de LSP: conviene revisar esa atribución al redactar; no se modificó el PDF.

### Datos, entrenamiento y líneas base

- GLUE: los scripts completos comentados enumeran MNLI, SST-2, MRPC, CoLA, QNLI, QQP y STS-B, pero el bucle activo ejecuta **solo MNLI**. `llm_finetuning.py` carga `nyu-mll/glue`; reserva los primeros 16 ejemplos para calibración y usa `validation_matched` en MNLI. El paper repite la etiqueta SST2 en Tabla 3: no corresponde inferir una octava tarea o reconstruir un promedio de siete como si fueran ocho.
- Alpaca: `data.py` lee `datasets/alpaca_gpt4_data.json`, procedente de `Instruction-Tuning-with-GPT-4/GPT-4-LLM`, no del JSON estándar de Stanford Alpaca. Usa un formato explícito instruction/input/output. El split es por orden: 90% entrenamiento, siguientes 16 ejemplos calibración y resto evaluación. Deben conservarse revisión del dataset, checksum, orden y tokenizer; la caché actual no distingue todas las opciones de preprocesamiento.
- Código: el dataset concreto está en los scripts de `simluation/example/`; no es un dataset local provisto. La ruta de calidad usa supervisión de respuestas y enmascarado del prompt. No se trasladará automáticamente a la ruta de Alpaca porque sus objetivos y collators difieren.
- Baselines originales: Full Parameter y GaLore(rango16) para GLUE; Zero-Offload y LoRA para Alpaca; Zero/Full Parameter, LoRA(rango8, alpha32) y GaLore(rango256, alpha0.25) para DeepSeek-1.3B. La ablación incluye Zero+planificación por capas y cómputo GPU sin comunicación. El repo principal implementa una ruta propia rotulada Zero; el flag `--use_deepspeed` no basta para afirmar que se ejecuta DeepSpeed.
- El paper selecciona para Alpaca LR 1e-4 en LSP y 1e-5 en LoRA/Zero; el script GPT-2 usa 1e-4 para Zero. LoRA aparece como rango16 en figura, rango32 en el lanzador GPT-2 y rango8 en lanzador Llama. Se deben declarar por separado configuraciones «script original» y «paper» y resolver la comparación antes de medir.
- Los scripts GLUE y Alpaca usan pesos **bfloat16**, con momentos float32 en los lanzadores. El texto general usa fp16 y Tabla 4 especifica bfloat16. No se supondrá que esas precisiones son intercambiables experimentalmente.

### Qué significa «tiempo real» en esta reproducción

El apéndice, p.12, explica que DeepSeek-Coder se entrena con memoria suficiente para obtener calidad por paso y luego se combina con un perfil del tiempo de offloading escalado al batch efectivo. Fig. 5c/d lo rotulan como simulado. Por tanto, los 120 h, 15 h y 30 h no demuestran una ejecución continua íntegra de ese fine-tuning con offloading en esos equipos.

Además, `scripts/draw_loss_gpt2_large.py` y su equivalente Llama reconstruyen el eje temporal a partir del índice y **segundos por iteración constantes**; no usan directamente los timestamps CSV. En GPT-2 incluso se fija el primer punto de perplexity a 2.5 y se escala una desviación de ventana por 0.15. No se reutilizarán estos scripts para afirmar tiempos propios o incertidumbre entre repeticiones. La reproducción local debe graficar timestamps efectivamente medidos y conservar evaluación/calibración en el tiempo total. No se ejecutarán gráficas sobre `result_ref/` como si fueran resultados nuevos.

## 3. Dependencias y compatibilidad

| Componente | Evidencia upstream | Propuesta para la laptop / condición |
| --- | --- | --- |
| Python | Binario publicado CPython 3.10; pyproject de la segunda ruta declara >=3.8, pero usa sintaxis más moderna | Python 3.10 o superior compatible con la imagen; registrar versión real; no usar el `.so` publicado |
| PyTorch/CUDA | No se fija versión reproducible del entorno completo | Conservar 2.7.1/CUDA12.8 previamente verificados por el usuario; revalidar extensión sm_120 y API |
| Transformers | Imports internos de modelos y entrenamiento; sin pin global | Resolver una versión compatible mediante prueba aislada y guardar lock; no afirmar un pin probado ahora |
| Datos/evaluación | `datasets`, `evaluate`, `diskcache`, `scikit-learn`, `numpy`, `pandas`, `tqdm` | Necesarios según ruta; fijar versiones y revisiones de datasets/modelos |
| Tokenización | AutoTokenizer; OpenLLaMA usa tokenizer lento | Verificar `tokenizers`, `sentencepiece` y sus dependencias cuando la ruta lo requiera |
| Compilación | `CUDAExtension` con C++, CPU y CUDA; `-fopenmp -mavx2` | Imagen de desarrollo con nvcc12.8, compilador C++ compatible, libgomp, setuptools/ninja; compilar con recursos limitados |
| Gráficas y telemetría | matplotlib en upstream; sin monitor integral | Añadir herramientas de monitorización separadas del método, con versiones registradas |
| Segunda ruta DeepSeek | tensorboard, PEFT/GaLore según baseline; fuerza `flash_attention_2`; API interna `check_target_module_exists` | Aislar y comprobar compatibilidad antes de usar; no necesaria para iniciar GPT-2/GLUE |
| DeepSpeed/Apex | No son importaciones requeridas por la ruta principal auditada | No copiar automáticamente las dependencias de STRONGHOLD |
| Otros | `ignite.metrics.Rouge` solo en rama Rouge; no usada en la ruta prioritaria | Instalar únicamente si se activa esa evaluación |

El upstream solo declara `torch` y `transformers` sin versiones en `simluation/pyproject.toml`; no publica Dockerfile ni lock global suficiente para reproducir exactamente su entorno. El comando del README `DEVICE=cpu ...` no evita CUDA: la selección por variable está comentada y se compilan los tres archivos. Hace falta un contenedor **devel**, no solo el runtime de STRONGHOLD. Comenzar con `MAX_JOBS=1` es una decisión de compilación, no un parámetro del método; limita procesos simultáneos, pero no garantiza que el pico de memoria quepa.

PyTorch documenta soporte Blackwell y binarios CUDA12.8 desde 2.7. Esto respalda la elección del entorno base, pero no demuestra que la extensión propia ni FlashAttention funcionen. NVIDIA documenta restricciones de memoria fijada y de algunas consultas NVML en WSL: se comprobará una transferencia fijada real y se registrarán métricas no disponibles como tales.

## 4. Presupuesto de memoria y CPU

Las especificaciones de la Lenovo son las comunicadas por el usuario: i7-13700HX, 16 núcleos/24 hilos, RTX5050 Laptop de 8 GB, capacidad12.0, RAM física16 GB y WSL2 con unos7.6 GiB en pruebas anteriores. No se midieron desde esta sesión. Los 300 GB de disco son capacidad de almacenamiento, no memoria para estados del optimizador.

LSP-Offload conserva los pesos base en GPU durante el entrenamiento. No es un sistema que permita guardar arbitrariamente pesos de un modelo 6.7B en SSD para hacerlo caber en 8 GB sin cambiar su ruta original. La calibración también mueve el modelo completo a CPU y conserva gradientes completos float32; por tanto, no basta con estimar el estado comprimido.

| Modelo nominal | Solo pesos de 16 bits, estimación decimal | Equivalente aproximado GiB | Juicio con 8 GB VRAM / 7.6 GiB WSL |
| --- | --- | --- | --- |
| RoBERTa 117 M del paper | 0.234 GB | 0.218 GiB | Holgura para comenzar; comprobar número real de parámetros del checkpoint, cabeza y batch de evaluación |
| GPT-2 Large 774 M | 1.548 GB | 1.442 GiB | Buen candidato, pero calibración/baseline pueden agotar RAM |
| DeepSeek 1.3B nominal | 2.600 GB | 2.421 GiB | Pesos caben; fine-tuning íntegro condicionado a picos CPU/GPU |
| OpenLLaMA 3B nominal | 6.000 GB | 5.588 GiB | Margen pequeño para activaciones/temporales; no prioridad inicial |
| DeepSeek 6.7B nominal | 13.400 GB | 12.480 GiB | Los pesos solos exceden VRAM; descartado en ruta original |

Son cotas de pesos, no pronósticos de pico. En especial, el nombre de RoBERTa y la cifra de parámetros del paper no sustituyen el conteo real. No incluyen allocator, activaciones, gradientes, buffers fijados, optimizadores, logits, calibración, carga del checkpoint ni duplicación temporal.

En la ruta principal, por matriz comprimida con momentos float32 hay al menos dos estados y un buffer de comunicación: aproximadamente `12*d*d` bytes en CPU, antes de copias adicionales. Con d512 son3 MiB por matriz; d1280 son18.75 MiB; d2048 son48 MiB. Para GPT-2 Large, 36 bloques con cuatro matrices seleccionadas por bloque dan144 matrices y unos432 MiB para esos tres tensores comprimidos. Se debe verificar la lista real de módulos, no convertir este cálculo en una medición.

La RAM de calibración puede ser mucho mayor: `_grad` se guarda en CPU float32 por cada matriz, el modelo se mueve a CPU, hay buffers/cargas y optimización de proyectores. En GPT-2, usar todos los774 M como cota ilustrativa da hasta3.096 GB de gradientes float32 más1.548 GB de pesos, antes del resto. El baseline no comprimido puede requerir aproximadamente12 bytes por parámetro seleccionado solo en momentos/buffer; por eso la comparación Zero puede fallar aunque LSP quepa. Si ocurre, se reportará la inviabilidad de ese baseline al tamaño original; no se reemplazará sin declararlo.

Los proyectores se almacenan como índices/valores, pero `get_sketch()` materializa matrices densas transitorias. No debe extrapolarse únicamente el coste disperso teórico. La recalibración puede mover momentos a GPU y elevar su pico.

La CPU i7 debe soportar la vía AVX2/OpenMP, pendiente de comprobar flags y cuotas efectivas de WSL/contenedor. El código usa por defecto14 hilos intra-op y2 inter-op: se registrarán y evaluará su contención con OpenMP en el equipo real. Tener24 hilos lógicos no garantiza rendimiento equivalente a Threadripper ni el ancho de banda PCIe publicado. Se medirán las dos direcciones y el solapamiento con buffers fijados, sin asumir10-20 GB/s del paper.

No se prevé descargar toda la colección de modelos ni guardar un checkpoint por iteración. Debe establecerse una política de retención antes de entrenar; en esta etapa no se borró ningún resultado ni se cambió la RAM asignada a WSL.

## 5. Plan de implementación con condiciones de avance

| Etapa | Acción concreta | Evidencia para avanzar |
| --- | --- | --- |
| A: completada documentalmente | Fijar commit, mapa paper/código, memoria, incompatibilidades y protocolo | Manifiesto y revisión de fuentes; ninguna afirmación de entrenamiento |
| B: pendiente en laptop | Contenedor exclusivo con PyTorch2.7.1/CUDA12.8 de desarrollo; importar dependencias; reconstruir extensión; medir capacidad real | Inventario de sistema, imagen/digest, lock, build log, kernel CPU/GPU y transferencia fijada |
| C1 | Validar kernel fusionado y transformación de momentos con tensores pequeños; validar compresor original sin sustituirlo | Errores absolutos/relativos y criterio explícito; CPU/GPU epsilon discrepante resuelto documentalmente |
| C2 | RoBERTa + MRPC o SST-2 de GLUE como diagnóstico corto; luego MNLI original si cabe | Pérdidas finitas, aprendizaje de valores de proyectores, recalibración y persistencia; no llamar a este diagnóstico Tabla3 completa |
| C3 | Comparar versión serial y planificador original FCFS/LCFS sobre mismos pesos, datos y proyectores | Actualización antes de forward, sin actualizaciones perdidas; trazas de streams y diferencias numéricas acotadas |
| D | GPT-2 Large + JSON Alpaca GPT-4 de los autores, primero pocas iteraciones y luego atravesar al menos una recalibración real | Sin OOM CPU/GPU, VRAM/RAM medidas y calidad evaluada; preservar d512 y probar primero r8 |
| E | LSP, baseline Zero del código y LoRA con configuración declarada; ablación de planificador; repeticiones | CSV/JSON/trazas y tiempos reales; el baseline se marca inviable si no cabe |
| F | Tablas/gráficas IEEE/LaTeX desde mediciones aceptadas | Scripts de análisis con procedencia y unidades; sin números de referencia mezclados |

Para GPT-2 se intentará batch4 y longitud512 del lanzador. Si el diagnóstico OOM obliga a batch1/2, se documentará como adaptación experimental y se aplicará una comparación coherente a las rutas. Reducir batch cambia la dinámica de entrenamiento: no constituye reproducción exacta del resultado publicado. No se recortarán capas ni se cambiará de familia de modelo sin declarar el cambio. Un GPT-2 menor puede servir después para diagnóstico si el baseline no cabe, separado del objetivo original.

La primera validación GLUE con `--offload cuda:0` conserva el lanzador del autor pero **no cierra la validación del sistema**. Una prueba posterior con `--offload cpu` se registrará como adaptación de configuración. Los experimentos completos se habilitarán únicamente después de resolver los bloqueos científicos de COMPATIBILIDAD.md.

## 6. Límites pendientes y acceso que falta

Se puede trabajar directamente sobre GitHub; no se dispone de un canal hacia `C:\PFC1`, la distribución Ubuntu de WSL2, Docker Desktop ni la RTX5050 de la laptop. El entorno de revisión no expone Docker CLI ni `nvidia-smi`. Falta ejecutar el diagnóstico dentro de esa laptop o conectar explícitamente un entorno de ejecución con su carpeta, Docker y GPU. Acceso de escritura GitHub no proporciona acceso a esos recursos.

Por ello no se puede afirmar aún una combinación final de versiones Transformers/extensión, que cabe un batch concreto, el consumo real de RAM, velocidad, calidad o solapamiento. La conclusión es una evaluación de viabilidad fundada y un orden de pruebas, no una reproducción experimental completada.

## Fuentes

1. Chen, S.; Wang, Z.; Guan, Z.; Liu, Y.; Gibbons, P. B. (2025). *Practical Offloading for Fine-Tuning LLM on Commodity GPU via Learned Sparse Projectors*. DOI: https://doi.org/10.1609/aaai.v39i22.34531. PDF adjunto: método pp.4-6; evaluación pp.6-7; implementación/GLUE/Alpaca p.11; DeepSeek y simulación p.12.
2. Código fijado: https://github.com/gulang2019/LSP-Offload/tree/d9dc92a32b7e4829116ba6abec17c280dfb548a4. Rutas y hashes en `original/UPSTREAM.json`.
3. PyTorch 2.7 y Blackwell/CUDA12.8: https://pytorch.org/blog/pytorch-2-7/.
4. PyTorch2.7.1, binarios CUDA12.8: https://pytorch.org/get-started/previous-versions/.
5. NVIDIA, restricciones WSL y memoria fijada: https://docs.nvidia.com/cuda/wsl-user-guide/. Consulta:12 de noviembre de2026; no implica migrar a CUDA13.
6. PFC1 inspeccionado: https://github.com/juanvt20/PFC1/tree/0b70b6a8a8572fa6765c2a4a616520763e6e922d. PDF PFC adjunto y hardware comunicado por el usuario: contexto, no telemetría nueva.
