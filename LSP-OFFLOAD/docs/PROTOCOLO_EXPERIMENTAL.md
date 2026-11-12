# Protocolo previsto: mediciones y reproducibilidad

Estado: diseñado, no ejecutado. Las cifras de rendimiento permanecerán vacías hasta medir en la laptop. No se han producido gráficas ni resultados propios en Etapa A.

## Condiciones de aceptación

1. Registrar CPU, flags AVX2, RAM física, MemAvailable WSL, límites cgroup, swap, espacio libre, GPU, VRAM disponible, compute capability, controlador Windows, versiones WSL/Docker y paquetes Python. Distinguir datos no disponibles de ceros. Los datos del usuario son contexto, no sustituyen este registro.
2. Guardar commit PFC1, commit upstream, diff de adaptación, revisión de modelos/datasets/tokenizador, hash de archivos, imagen y digest, compilador/nvcc, comandos y variables no secretas. No volcar el entorno completo porque puede contener tokens.
3. Validar la extensión original en CPU float32 y GPU, con su fórmula concreta, momentos, error absoluto/relativo y tamaños pequeños. La discrepancia de epsilon debe resolverse antes de afirmar equivalencia; una sustitución por AdamW u otro optimizador no es compatibilidad.
4. Validar dimensiones, gradientes finitos, aprendizaje de valores de proyectores, error de reconstrucción, actualización de subespacio y transformación de ambos momentos. El primer fit de la ruta principal no demuestra aprendizaje; forzar un diagnóstico que alcance recalibración y declararlo como diagnóstico.
5. Comprobar el orden de actualización en modo serial y FCFS/LCFS, fin de época, evaluación y checkpoint. Usar los mismos pesos, minibatches y proyectores al estudiar únicamente la planificación. No usar una trayectoria full-rank como referencia de igualdad exacta para un método comprimido: la compresión es deliberadamente aproximada.
6. Probar que un checkpoint reanuda el paso siguiente con modelo, momentos, proyectores/índices, contadores, RNG, orden de datos y scheduler coherentes. El guardado upstream del modelo solo no satisface este requisito.
7. Hacer un diagnóstico corto antes de cada fine-tuning; después cruzar al menos un hito real de recalibración. Solo después lanzar presupuestos completos.

## Artefactos por ejecución

Identificador sugerido: fecha UTC + modelo + tarea + método + semilla, sin sobrescribir ejecuciones anteriores. Mantener hora local para el usuario y UTC en registros.

| Archivo | Contenido requerido |
| --- | --- |
| `config.json` | Parámetros efectivos, distinción paper/script/adaptación, módulos entrenados y congelados, batch real/efectivo, precisión de pesos/estados, d,r, frecuencia/umbral, semillas, hilos y planificador |
| `environment.json` | Hardware, límites WSL/cgroup, versiones, imagen/digest y revisiones |
| `command.txt`, `stdout.log`, `stderr.log` | Comando exacto con secretos ocultos, salida completa y errores |
| `samples.csv` | Telemetría temporal por proceso/árbol/cgroup, VRAM, RAM, swap, CPU y métricas GPU disponibles |
| `steps.csv` | Paso/época, ejemplos y tokens efectivos, pérdida bruta, LR, duración y fases cuando medibles |
| `evaluations.csv` | Puntuación bruta por tarea y checkpoint, duración y tamaño real de evaluación |
| `summary.json` | Estado success/failed/OOM/interrupted, exit code, duración, picos, throughput, calidad, errores y rutas |
| `trace/` | Eventos CUDA/NVTX u otra traza para comprobar solapamiento; solo en corridas dedicadas |
| `checkpoints/` | Estado completo cuando se requiera reanudación; excluido de Git |

El monitor y el resumen deben guardar salida también ante excepción u OOM cuando el proceso supervisor siga vivo. Un kill total de WSL puede impedir escribir el resumen final: conservar mediciones incrementalmente y permitir reconstruir el estado incompleto. Nunca etiquetar un resultado fallido como cero consumo o cero tiempo.

## Métricas y comparaciones

| Dimensión | Métrica / interpretación |
| --- | --- |
| Rendimiento original | Iteraciones/s, segundos/iteración y slowdown frente a cómputo GPU cuando proceda |
| Rendimiento adicional | Muestras/s y tokens/s reales; distinguir padding de tokens válidos y microbatch de actualización |
| Tiempo total | Wall-clock de la ejecución completa, con carga/calibración/evaluación/checkpoint separados; no excluir silenciosamente esas fases |
| GPU | Máximos allocated/reserved de PyTorch y memoria de dispositivo consultable; son medidas distintas |
| RAM | RSS del proceso/árbol, cgroup y MemAvailable WSL; buffers fijados y swap cuando medibles. No sumar dos veces procesos compartidos |
| Comunicación | Bytes D2H/H2D reales, tiempo por dirección, esperas y traza de solapamiento; ancho de banda con unidades definidas |
| GLUE | MNLI/SST2/QNLI: accuracy; CoLA: Matthews; MRPC/QQP: accuracy y F1; STS-B: Pearson y Spearman. Guardar todos los campos que devuelve el evaluador |
| Alpaca | Perplexity y pérdida de evaluación con el mismo split/collator; menor es mejor. No llamar accuracy a perplexity |
| Código | Loss y HumanEval multilenguaje solo con harness y protocolo fijados; no inferir pass@k del nombre de la tabla |
| Compresor | Error relativo en calibración y validación, tiempo de fit, veces que se recalibra y dimensión efectiva |

Primero se comparan rutas al mismo modelo, datos, secuencia, precisión y batch viable. Se informará aparte el mayor batch que cabe en cada método. Para calidad, comparar presupuesto de tiempo y pasos/tokens; los LR de los autores pueden diferir por método y deben declararse.

La primera ejecución es diagnóstica, sin inferencia estadística. Para afirmaciones de rendimiento/variabilidad, planificar al menos tres repeticiones independientes si el coste lo permite, guardar semillas y reportar mediana/dispersión. Para calidad, variar también semillas cuando sea viable. La desviación de una media móvil no es un intervalo entre ejecuciones. Un número menor de repeticiones debe quedar como limitación, no convertirse en significancia estadística.

## Gráficas y redacción

Los scripts propios usarán únicamente CSV de ejecuciones identificadas. Mantendrán datos brutos y parámetros de suavizado. No se modificará el primer punto ni se sustituirá el tiempo medido por constantes de los autores. Toda estimación futura se rotulará como estimación y se separará de mediciones. Exportar PDF/PNG y tablas LaTeX IEEE con unidades y referencias a la configuración.

La mejora original de33.1%-62.5% no es una predicción para esta laptop. Tampoco se compararán directamente los resultados de preentrenamiento STRONGHOLD con este fine-tuning para afirmar superioridad de una técnica.
