# PFC1: LSP-Offload

Estado actual: **main-v1 se detuvo en Zero tras165 pasos por una aserción de NaN; LSP no se inició**. El diagnóstico acotado confirma una lectura de delta antes de esperar H2D y propone el parche0003, probado únicamente en copia diagnóstica. Los replays no repitieron el NaN histórico, por lo que no se atribuye su causa exclusiva sin evidencia. Se conservan los pilotos y las seis repeticiones previas. Véanse [diagnóstico concreto](configs/zero_nan_diagnosis.json), [registro de ejecuciones](docs/VALIDACION_LOCAL.md) y [comparación corta](configs/comparison_summary.json).

La primera reproducción propuesta utiliza **RoBERTa-base + GLUE** para validar el aprendizaje de proyectores y **GPT-2 Large + Alpaca GPT-4** para validar offloading CPU y planificación por capas. DeepSeek-Coder-1.3B queda condicionado a memoria y corrección; 6.7B no cabe en la ruta original con 8 GB de VRAM.

- [Evaluación técnica completa](docs/ETAPA_A_VIABILIDAD.md).
- [Hallazgos de compatibilidad y fidelidad](docs/COMPATIBILIDAD.md).
- [Protocolo de medición y criterios de avance](docs/PROTOCOLO_EXPERIMENTAL.md).
- [Matriz trazable de experimentos, métricas y viabilidad](docs/MATRIZ_EXPERIMENTOS.md).
- [Contraste de PDF, código y evaluación previa](docs/CONTRASTE_PDF_CODIGO.md).
- [Acceso local y validación del entorno](docs/VALIDACION_LOCAL.md).
- [Herramientas PDF aisladas y comandos](docs/HERRAMIENTAS_PDF.md).
- [Procedencia del código original](original/UPSTREAM.json).
- [Plan legible por herramientas](configs/etapa_a_plan.json).

El upstream auditado es https://github.com/gulang2019/LSP-Offload, commit `d9dc92a32b7e4829116ba6abec17c280dfb548a4`. Su copia completa está en `original/upstream/`, excluida de Git; sus 99 archivos coinciden byte por byte con el manifiesto. No se redistribuye código de terceros sin licencia explícita identificada. Los PDF se conservan locales en `referencias/`, también excluido.

## Organización

| Directorio | Uso |
| --- | --- |
| `original/` | Procedencia y copia intacta en `original/upstream/`, excluida de Git |
| `adaptado/` | Copia local excluida de Git, parches separados N15/N14 y pruebas versionadas |
| `docker/` | Dockerfile independiente, requisitos y registro de versiones; imagen local `pfc-lsp-offload:torch271-cu128` |
| `configs/` | Planes y configuraciones versionados |
| `datasets/` | Datos y cachés locales, excluidos de Git |
| `models/` | Pesos y revisiones descargadas, excluidos de Git |
| `results/` | Ejecuciones y checkpoints locales, excluidos de Git |
| `logs/` | Logs de ejecución locales, excluidos de Git |
| `metrics/` | Telemetría y trazas locales, excluidas de Git |
| `docs/` | Evaluación, compatibilidad y protocolo |

No se modifican archivos de `STRONHOLD/`. El trabajo existente de STRONGHOLD y el nuevo ajuste fino de LSP-Offload son evaluaciones independientes: difieren en modelo, datos y objetivo, por lo que sus números no forman una comparación directa de velocidad.

## Acceso efectivo

Actualización local del 12 de noviembre: lectura de `C:\PFC1`, Git, Docker
y operación CUDA con PyTorch 2.7.1/CUDA 12.8 confirmadas. También se
compiló la extensión original y se probaron imports, kernel float32 y
transferencia fijada pequeña en la imagen independiente. Véanse las
mediciones y límites en [Validación local](docs/VALIDACION_LOCAL.md).
El párrafo siguiente describe únicamente el entorno de la revisión anterior.

GitHub es accesible y el repositorio PFC1 es público. El entorno de esta revisión no está conectado al disco `C:\PFC1`, a WSL2 ni a Docker Desktop de la laptop. La GPU y las versiones locales informadas por el usuario no se verificaron nuevamente. No se instalaron componentes ni se descargaron modelos o datasets.
