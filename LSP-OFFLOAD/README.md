# PFC1: LSP-Offload

Estado al 12 de noviembre de 2026: **Etapa A completada como revisión documental y estática**. La ejecución en la laptop está pendiente. No hay resultados de entrenamiento propios en esta entrega.

La primera reproducción propuesta utiliza **RoBERTa-base + GLUE** para validar el aprendizaje de proyectores y **GPT-2 Large + Alpaca GPT-4** para validar offloading CPU y planificación por capas. DeepSeek-Coder-1.3B queda condicionado a memoria y corrección; 6.7B no cabe en la ruta original con 8 GB de VRAM.

- [Evaluación técnica completa](docs/ETAPA_A_VIABILIDAD.md).
- [Hallazgos de compatibilidad y fidelidad](docs/COMPATIBILIDAD.md).
- [Protocolo de medición y criterios de avance](docs/PROTOCOLO_EXPERIMENTAL.md).
- [Procedencia del código original](original/UPSTREAM.json).
- [Plan legible por herramientas](configs/etapa_a_plan.json).

El upstream auditado es https://github.com/gulang2019/LSP-Offload, commit `d9dc92a32b7e4829116ba6abec17c280dfb548a4`. Se debe conservar una copia completa, sin cambios, antes de adaptar código. Esta entrega fija su procedencia; no redistribuye código de terceros sin licencia explícita identificada.

## Organización

| Directorio | Uso |
| --- | --- |
| `original/` | Procedencia; futura copia intacta en `original/upstream/`, excluida de Git |
| `adaptado/` | Futura copia de trabajo, parches versionados y registro de diferencias |
| `docker/` | Diseño del contenedor aislado; construcción pendiente de Etapa B |
| `configs/` | Planes y configuraciones versionados |
| `datasets/` | Datos y cachés locales, excluidos de Git |
| `models/` | Pesos y revisiones descargadas, excluidos de Git |
| `results/` | Ejecuciones y checkpoints locales, excluidos de Git |
| `logs/` | Logs de ejecución locales, excluidos de Git |
| `metrics/` | Telemetría y trazas locales, excluidas de Git |
| `docs/` | Evaluación, compatibilidad y protocolo |

No se modifican archivos de `STRONHOLD/`. El trabajo existente de STRONGHOLD y el nuevo ajuste fino de LSP-Offload son evaluaciones independientes: difieren en modelo, datos y objetivo, por lo que sus números no forman una comparación directa de velocidad.

## Acceso efectivo

GitHub es accesible y el repositorio PFC1 es público. El entorno de esta revisión no está conectado al disco `C:\PFC1`, a WSL2 ni a Docker Desktop de la laptop. La GPU y las versiones locales informadas por el usuario no se verificaron nuevamente. No se instalaron componentes ni se descargaron modelos o datasets.
