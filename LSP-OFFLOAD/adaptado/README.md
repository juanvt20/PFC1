# Adaptación aislada: momentos y cierre del planificador

`patches/0001-preserve-transported-moments.patch` es el primer cambio aplicado
al upstream en la copia local `upstream/`, excluida de Git. Corrige N15 con
la autorización del usuario: conserva M/V ya transportados al preparar
buffers y los devuelve al dispositivo del optimizador. No reinicia su
contador. La primera inicialización sigue creando ceros. Solo el compresor
LearnedCountSketch activa la nueva opción; la ruta SVD no se modifica.

Referencia: PDF LSP p.5, Algoritmo 1, líneas 8-9. El código almacena las
transpuestas de P/Q. Si A=P_nuevo^T P_anterior y B=Q_nuevo^T Q_anterior:
`M_nuevo=A M_anterior B^T`, `V_nuevo=A² V_anterior (B^T)²`, con cuadrados
elemento a elemento. La fórmula existente permanece intacta, igual que
epsilon, regularización, inicialización, muestreo y tratamiento de no finitos.

Reconstrucción desde `C:\PFC1`:

```powershell
python LSP-OFFLOAD/tools/prepare_adaptation.py
```

El preparador verifica los 99 blobs del manifiesto antes de copiar;
rechaza sobrescribir una copia adaptada que difiera de su salida esperada.
La copia original no se escribe. El parche también admite comprobación
con `git apply --check` desde el upstream usando su ruta absoluta.

Pruebas reproducibles en la imagen aislada existente:

```powershell
& ./LSP-OFFLOAD/tools/run_regression.ps1 -Suite moments -Source original -RunId moment-original-captured-inputs
& ./LSP-OFFLOAD/tools/run_regression.ps1 -Suite moments -Source adaptado -RunId moment-adapted-captured-inputs
& ./LSP-OFFLOAD/tools/run_regression.ps1 -Suite scheduler -Source adaptado -RunId scheduler-boundaries-before-fix
```

Esos IDs ya existen: usar otros para repetir sin sobrescribir evidencia.
La primera prueba falla como se espera en el original (exit1); la segunda
pasa (exit0). Verifica los tensores completos, dos recalibraciones,
inicialización, buffers y siguiente paso de Adam: CPU/GPU y pesos
float32/bfloat16, estados float32. Captura los sketches densos realmente
usados; no los regenera para evitar que colisiones cambien la referencia.
Referencia de transporte en float64 sobre los solapamientos con el redondeo
original; tolerancias rtol2e-5/atol2e-6. Los tensores completos se guardan
localmente en `.pt`; el resumen está en `configs/adaptation_validation.json`.

La prueba anterior del planificador (`scheduler-boundaries-before-fix`)
confirma N14. Con la autorización posterior se aplica por separado
`patches/0002-drain-pending-operations.patch`. `tools/scheduler_patch.py`
construye ese parche sobre el primero, conservando sus bytes.

`Scheduler.drain()` consume callbacks en el orden del siguiente forward y
espera CUDA antes de evaluar, recalibrar, guardar o salir normalmente.
`clear()` rechaza trabajo pendiente; un drain fallido no puede reintentarse
silenciosamente. No se vacía la cola en cada iteración. Se conservan FCFS,
LCFS y sus delays. Los hooks que aplican pesos se ejecutan antes del matmul
que los consume (Algoritmo3, p.11, línea5), y `record_stream` protege los
buffers CUDA usados por D2H y actualización frente a reutilización temprana.
Estas dos dependencias también forman parte del parche de planificación;
no cambian fórmulas ni añaden sincronizaciones por iteración.

`scheduler-drain-multistep-v3` pasa los ocho casos: Zero/LSP, con/sin
checkpointing, pesos float32/bfloat16 y Adam CPU float32. Son 12 pasos y
seis capas con entradas, pesos iniciales y proyectores iguales al control
serial; W/M/V completos y contadores coinciden exactamente en esta ejecución
(tolerancia admitida atol2e-6/rtol5e-5). Incluye evaluación, recalibración
real, guardado, salida, doble vaciado y rechazo de clear sin descartar trabajo.
Los proyectores d8/r1 de esta prueba son diagnósticos, no el perfil d512/r8.
Se conservan los fallos v1/v2 y sus parches para explicar las correcciones.

`scheduler-gpt2-checkpointing-v2` confirma en GPT2 pequeño aleatorio que
congelar embeddings exige `enable_input_require_grads()` para el checkpointing
reentrante de esta versión de Transformers. El wrapper propio lo activa
para ambas rutas sin entrenar embeddings; las ocho matrices reciben seis
actualizaciones. El control sin hook falla como se esperaba. El launcher
upstream no incorpora ese hook: se documenta como adaptación del wrapper.

El piloto y la comparación se ejecutan con `tools/run_comparison.ps1`,
configuración `configs/short_comparison.json` y activos fijados en
`configs/pilot_assets.json`. No usan los nombres de scripts como identidad:
Zero tiene `Scheduler(zero=True)` y compresor identidad; LSP tiene
`zero=False`, CountSketch y planificación FCFS/LCFS explícita. Estado y
evidencia actualizados en `docs/VALIDACION_LOCAL.md`.

Tras el fallo de main-v1 se prepara, **sin aplicar a esta copia**, la propuesta
`propuestas/0003-wait-before-validating-delta.patch`: esperar el evento H2D
antes de leer delta y conservar una aserción reforzada con isfinite. No está
en `patches/` ni se incorpora al preparador, para distinguirla de los dos
parches activos. Prueba dirigida y replay limitado a166 pasos en
`configs/zero_nan_diagnosis.json`; no habilita un relanzamiento largo.
