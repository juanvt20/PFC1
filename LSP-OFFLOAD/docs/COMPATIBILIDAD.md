# Registro de compatibilidad y fidelidad

Fecha: 2026-11-12. Todos los hallazgos proceden del commit upstream `d9dc92a32b7e4829116ba6abec17c280dfb548a4`. **No se ha aplicado ningún parche al código de los autores.** «Pendiente» significa que el comportamiento se deduce de las fuentes, sin validación de ejecución en la laptop.

## Clasificación de cambios futuros

- Compatibilidad: construcción, importaciones o API, preservando las operaciones matemáticas y dependencias de ejecución.
- Configuración experimental: batch, longitud, tamaño de datos, semilla, número de hilos, destinos CPU/GPU y parámetros del planificador. Deben quedar explícitos aun si el método no cambia.
- Instrumentación: métricas, manifiestos, temporizadores, trazas y persistencia. No debe alterar la actualización del modelo.
- Corrección con efecto numérico o científico: optimizador, proyectores, reconstrucción, estados, selección de checkpoints o etiquetas experimentales. Se conserva upstream y se documenta el efecto antes de usar resultados.

## Bloqueos antes del entrenamiento aceptado

| ID | Evidencia reproducible | Implicación / acción pendiente | Clase |
| --- | --- | --- | --- |
| C01 | `kernels/setup.py:5-8,13-21`: selección DEVICE comentada; CUDAExtension siempre incluye CPU/CUDA | `DEVICE=cpu` no elimina nvcc; reconstruir en imagen devel CUDA12.8. No importar el `.so` CPython3.10 del repo | Compatibilidad |
| C02 | `simluation/pyproject.toml`: solo torch/transformers sin pins; no lock global | Resolver dependencias en aislamiento, verificar imports internos y congelar versiones/digest. No asignar versiones «originales» desconocidas | Compatibilidad |
| C03 | README usa `cd simulation`; árbol contiene `simluation/` | Usar ruta real en wrappers propios y conservar copia intacta | Compatibilidad de lanzador |
| C04 | `llm_finetuning.py:656-659` y kernel CPU | Mantener optimizador float32 en CPU; kernel accede `data_ptr<float>()` y exige numel múltiplo de8. Comprobar AVX2, contigüidad y tamaños | Compatibilidad / precondición |
| N01 | `kernels/fusedAdam_cpu.cpp:35-36` frente a `fusedAdam_cuda.cu:23-25` | CPU calcula denominador `sqrt(v_hat+eps)`; GPU, `sqrt(v_hat)+eps`. **No son el mismo Adam.** No sustituir silenciosamente por torch.optim.Adam ni declarar equivalencia CPU/GPU. Probar la fórmula exacta de cada ruta y decidir/documentar la corrección numérica separadamente | Corrección matemática pendiente |
| N02 | `gradient_compressor.py:268-279`; segunda ruta en `simluation/lspoffload/compressor/count_sketch.py:43-76` | Principal usa randint con reemplazo: puede repetir posiciones; scatter_ no suma colisiones. Segunda ruta usa muestreo sin reemplazo. No fusionarlas ni garantizar exactamente r no ceros por fila en la primera | Diferencia de implementación del método |
| N03 | `gradient_compressor.py:271` | Inicialización binaria principal usa `randint(0,1)`, que produce ceros y luego valores negativos; segunda ruta usa `(0,2)`. Scripts principales usan gaussiana por defecto. Registrar y evitar activar la binaria como si fuera equivalente sin resolverlo | Corrección matemática pendiente |
| N04 | `LearnedCountSketchCompressor.train/fit`, `gradient_compressor.py:315-454` | El código optimiza error relativo y no añade explícitamente el regularizador beta del pseudocódigo. Primer fit crea proyectores sin aprender con datos; el aprendizaje efectivo ocurre en recalibraciones posteriores. No añadir el término ni afirmar aprendizaje validado en una prueba que no atraviesa ese hito | Diferencia paper/código |
| N05 | `llm_finetuning.py:690` y scripts Alpaca | Default `compress-cs-thresh=0.8`, frente a0.5 del paper. GLUE sí fija0.3. La actualización se comprueba cada1000 pasos locales y también al empezar época por reinicio del contador | Configuración experimental |
| N06 | `scripts/run-gpt2-large.sh`, `scripts/run-llama-3b.sh` y `llm_finetuning.py:663-669,1029-1036` | LSP no fija parámetros de planificación; defaults son0. El lanzador rotulado Zero no activa `--scheduler-zero`. Ejecutar esos scripts no demuestra la ablación FCFS/LCFS del paper. Hay que definir y verificar explícitamente el plan y la línea base | Configuración / identidad de baseline |
| N07 | Scripts GPT-2 y Llama, LR/rank de LoRA; paper apéndice p.11 | Discrepancias LR y rangos. Mantener perfiles «script» y «paper» separados; no retocar solo un competidor para mejorar una comparación | Configuración / comparabilidad |
| N08 | `llm_finetuning.py:907-910,991-998`, sustitución de módulos en `para_offload_modules.py` | Cuando hay offload se congelan parámetros ajenos a las matrices seleccionadas, excepto clasificador. `FullParam` y `Zero` son etiquetas upstream, no prueba de que se actualizan todos los parámetros. Publicar nombres/conteo reales por ruta | Identidad de baseline |
| N09 | `llm_finetuning.py:204,222` | Selecciona `score > best_score` también para perplexity: favorecería el peor checkpoint. Guardar puntuaciones brutas; una corrección de selección debe registrarse con efecto en calidad | Corrección de evaluación |
| N10 | `llm_finetuning.py:222,297-313` | Solo state_dict del modelo y CSV; no todos los momentos, proyectores, RNG ni cursor de datos. `--from-ckpt` no demuestra continuación exacta de LSP | Persistencia / reproducibilidad |
| N11 | `llm_finetuning.py:670,712-713`; scripts principales sin seed | No se fuerza semilla por defecto, pese a la afirmación del apéndice. Elegir y registrar semillas locales; no presentarlas como semillas de autores conocidas | Configuración / reproducibilidad |
| N12 | `data.py:14-20`; clave de caché de `InstructionTuningDataset` | Caché no incluye longitud, revisión y todas las decisiones del tokenizador. Aislar por configuración antes de variar secuencias; conservar hashes/splits | Preparación de datos |
| N13 | `kernels/fusedAdam_cuda.cu:64`, lanzamiento sin stream explícito | Validar comportamiento con streams CUDA actuales de PyTorch2.7.1 y el planificador. No asumir que una operación correcta aislada es correcta concurrentemente | Compatibilidad de concurrencia |
| N14 | `llm_finetuning.py:197`, `Scheduler.clear`, hooks de actualización | Se limpian hooks en evaluación y fin de época. Comprobar que no se descartan actualizaciones pendientes antes de evaluar/guardar; el análisis estático no demuestra que ocurra un error concreto | Corrección de ejecución pendiente |

Los números de línea son orientativos para esta revisión; el identificador decisivo es el commit y el símbolo de código. Los parches futuros deben incluir diff y validación en lugar de reemplazar archivos originales.

## Resultados que no deben inferirse de los scripts

| Hallazgo | Regla de reproducción |
| --- | --- |
| `kernels/test.py` define `test_correctness` pero sus invocaciones están comentadas; al ejecutarlo lanza benchmarks con tensores grandes | No usar su ejecución como evidencia de corrección. Validar con tamaños pequeños y tolerancias explícitas |
| `para_offload_modules.py` en main usa `torch.randint` para tokens y ajustes de compresores sin calibración de calidad | Etiquetar como microbenchmark medido con entradas sintéticas, nunca como fine-tuning real ni evaluación de calidad |
| Fig.5c/d y apéndice p.12 combinan calidad en GPU suficiente y perfil de offloading | No construir curvas de entrenamiento local continuo con esos datos ni heredar las constantes temporales |
| Scripts de gráficas reconstruyen tiempo; el entrenador excluye tiempo de evaluación de su columna `time` al ajustar `start_time` | Medir wall-clock externo incluyendo carga/calibración/evaluación y conservar además tiempos por fase |
| Tabla3 repite SST2 y el bucle GLUE activo solo ejecuta MNLI | Listar tareas realmente medidas, sin repetir ni atribuir el promedio original a otra selección |
| No se identificó pipeline completo de HumanEval/MultiPL-E ni configuración de generación en el árbol | No prometer reproducir sus scores hasta fijar harness/revisión, lenguajes, generación y métrica. Los tests de código deben ejecutarse aislados |
| No hay archivo LICENSE/COPYING en el árbol ni licencia explícita general identificada | Mantener atribución, copia local intacta y referencia por commit; no atribuir MIT/Apache ni republicar automáticamente el código adaptado. Los parches propios pueden versionarse por separado |

La ausencia de una licencia explícita identificada es un límite para redistribución, no una excusa para inventar otra técnica o dejar de estudiar el código publicado.
