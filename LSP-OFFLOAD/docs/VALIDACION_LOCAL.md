# Validación local: 12 de noviembre de 2026

## Alcance confirmado

Se confirmó lectura de `C:\PFC1` y ejecución de Git 2.56.0.windows.4.
Docker no está en el PATH de esta sesión. Su ejecutable está en
`C:\Users\juan\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe`.
El cliente y servidor son 29.9.2; Docker Desktop 4.98.0, contexto
`desktop-linux`. Las consultas WSL/Docker necesitaron salir del sandbox.
WSL informa `docker-desktop` en ejecución y `Ubuntu-ParlayANN` detenida.

Esto actualiza las limitaciones de acceso descritas en README y Etapa A:
aquellas describen la revisión estática anterior, no los accesos actuales.

## Diagnóstico de GPU y memoria

Se ejecutó la imagen local
`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` con
`--rm -i --pull=never --network none --gpus all`, sin montar carpetas.
No se instalaron paquetes. PyTorch informó `2.7.1+cu128`, CUDA `12.8`,
`torch.cuda.is_available() == True`, RTX 5050 Laptop GPU y capacidad `(12, 0)`.
La suma de dos tensores de un elemento en CUDA produjo `2.0`.
Este primer diagnóstico comprobó una operación básica. La ampliación con
kernel original y transferencia fijada se describe más abajo; concurrencia,
entrenamiento y calidad siguen sin validarse.

| Medición | Valor observado |
| --- | --- |
| VRAM libre / total, `torch.cuda.mem_get_info()` | 7,405,043,712 / 8,546,484,224 bytes |
| Linux MemAvailable / MemTotal | 6,720,492 / 7,979,996 KiB |
| Linux SwapFree / SwapTotal | 2,097,152 / 2,097,152 KiB |
| cgroup memory.max | `max` (sin límite adicional explícito) |
| cgroup memory.current | 1,439,948,800 bytes |
| Windows RAM libre / visible, consulta separada | 3,303,800 / 16,478,072 KiB |
| Driver Windows | 610.74 |

Son instantáneas de consultas sucesivas, no picos ni memoria garantizada
para entrenamiento. `memory.max=max` no elimina el límite de la VM WSL.

## Referencias locales

Los tres PDF se conservan en `referencias/`, excluido mediante
`LSP-OFFLOAD/.gitignore`; `git check-ignore -v` confirma los tres archivos.

| Archivo | Bytes | SHA-256 |
| --- | ---: | --- |
| LSP-OFFLOAD.pdf | 1083574 | f1bda470198ddfe30fcb68d0ae8b0d9881db263a0b45ada370800554195b55db |
| PFC (1).pdf | 226862 | a02537c6d6584334ded8489cd9bea1248ae8a77f99978a035f45854f2654be3b |
| STRONGHOLD.pdf | 886393 | 87cdc3327797b82dcba0e977fe02168180ad31f1a3516d645acdeb618a0d1a13 |

Se completó la lectura de LSP y después del PFC; STRONGHOLD se consultó
como contexto. La cobertura visual y la reconciliación por página/sección
y función están en [CONTRASTE_PDF_CODIGO.md](CONTRASTE_PDF_CODIGO.md).

## Herramientas PDF comprobadas

Python Windows 3.14.7: sin `pypdf`, `pdfplumber` ni `fitz` (PyMuPDF).
`pdftotext`, `pdftoppm` y `pdfinfo` no aparecen en el PATH de Windows.
En la imagen PyTorch de desarrollo tampoco se encontraron esos tres
módulos ni esos tres ejecutables. Tras autorización expresa se instaló
PyMuPDF1.26.7 en `.venv-pdf-linux/`, sin heredar paquetes globales.
Ubicación, versiones, intento MSYS2 fallido y comandos exactos en
[HERRAMIENTAS_PDF.md](HERRAMIENTAS_PDF.md).

No se modificó STRONHOLD ni se ejecutaron entrenamientos. Se ejecutó
`kernels/setup.py` para compilar las fuentes originales dentro de la imagen
independiente y se importaron módulos upstream para diagnósticos pequeños.

## Copia original

Se clonó el upstream en `original/upstream/` (excluido de Git del PFC),
en detached HEAD `d9dc92a32b7e4829116ba6abec17c280dfb548a4`.
La conversión CRLF del checkout Windows causó discrepancias de bytes;
`checkout-index --all --force` con `core.autocrlf=false` no las eliminó;
se materializaron los archivos afectados con los bytes exactos de
`git cat-file blob <SHA>`, sin cambiar configuración persistente.
La comprobación SHA-1 de blob Git sobre los bytes de los 99 archivos del
manifiesto terminó sin diferencias. Git sigue mostrando modificaciones
en archivos de texto incluso con `core.autocrlf=false` y refresco del
índice. Se identificó `core.autocrlf=true` heredado del gitconfig de sistema.
`git ls-files --eol -- gradient_compressor.py` informa `i/lf w/lf`,
`git hash-object --no-filters` y con filtros devuelven el hash del manifiesto,
y `git -c core.autocrlf=false diff --quiet` termina con código0. Son
avisos/estadísticas de checkout, no diferencias reales en las fuentes.
La comprobación de bytes se repitió después de las pruebas y pasó para
los 99 archivos. No se hizo normalización masiva ni cambios de configuración
global o local de Git; no se declara limpio el estado visual `status`.
No se importaron los binarios precompilados ni se usaron `result_ref/`
como mediciones propias.

## Entorno independiente y validación parcial

Imagen propia `pfc-lsp-offload:torch271-cu128`, construida con base por
digest (véase `docker/Dockerfile`). Python3.11.13, PyTorch2.7.1+cu128,
Transformers4.44.2, datasets2.21.0, evaluate0.4.3 y PyArrow17.0.0.
La primera resolución trajo PyArrow26, incompatible con NumPy1.26; ese
fallo se conservó y el perfil se corrigió exclusivamente en esta imagen.
Los 154 paquetes efectivos quedan registrados en
`configs/environment_packages.json`; restricciones resueltas en
`docker/constraints.txt`, identidad local en `configs/image_identity.txt`.
El freeze bruto incluye rutas de construcción conda de la base y es un
inventario, no un requirements portable para instalar desde cero.

Se compiló `CUDAExtension` sin parches, con `MAX_JOBS=1`, arquitectura12.0,
nvcc12.8.61 y g++11.4.0. `pip check` pasó. La detección de GPU se hace
en ejecución; el aviso “No CUDA runtime” durante build sin GPU no fue un
fallo de compilación. No se modificaron imágenes de otros proyectos.

Pruebas sintéticas offline, upstream montado de solo lectura:

| Prueba | Evidencia / límite |
| --- | --- |
| Importaciones | Transformers, datasets, evaluate, diskcache, sklearn y los módulos originales `gradient_compressor`, `para_offload_modules`, `llm_finetuning` importan correctamente. No se probaron descarga de datos ni todas las ramas del programa. |
| Kernel CPU y GPU | 32 elementos float32, tres pasos; se comparan delta y M/V con una referencia vectorizada de la fórmula específica de cada dispositivo. Error absoluto máximo observado menor que 3.5e-9, tolerancia2e-6. CPU/GPU siguen usando denominadores distintos (N01). |
| Memoria fijada | Copia H2D y D2H de 1MiB por dirección en stream explícito, sincronización y comparación exacta. No es una medida de ancho de banda ni prueba de solapamiento del planificador. |
| Compresor | Inicialización binaria con valor único -0.5; colisiones81/128 en configuración d=8,r=4; segundo fit cambia proyectores. Son comprobaciones diagnósticas, no entrenamiento ni estimación del caso d=512. |
| Transporte de momentos | Matriz16×16, d=8,r=2, float32, dos fits forzados con umbral -1 e instrumentación que observa antes/después sin cambiar cálculo. M/V transportados tienen suma absoluta24.344589 /216.153290 antes de `module.init`; ambos terminan en0. Confirma N15. |

`configs/environment_validation.json` conserva las medidas pequeñas y
declara `algorithm_accepted=false`. El exit code0 del diagnóstico significa
que se pudieron ejecutar sus comprobaciones de entorno/kernel, no que se
acepte el algoritmo. No se ejecutó entrenamiento largo, no se descargaron
modelos/datasets y no hay mediciones locales de calidad o throughput.

Los logs y los JSON/freeze detallados permanecen en `logs/` y `results/`,
excluidos de Git; el resumen e inventario sí son versionables. Las ramas
de planificador concurrente, cierre de hooks, precisión bf16, reanudación
completa y fin de época requieren pruebas posteriores. Cambiar epsilon,
regularización, inicialización o conservación de momentos requiere decisión
explícita del usuario antes de aplicar parches en una copia adaptada.

## Adaptación autorizada y bloqueo de ejecución: 13 de noviembre de 2026

El usuario autorizó exclusivamente conservar los momentos después de su
transporte. Se comprobó visualmente el Algoritmo1 p.5, líneas8-9, y se
reemplazó la evidencia basada solo en sumas por una prueba de todos los
elementos. Parche: `adaptado/patches/0001-preserve-transported-moments.patch`.
Solo cambia la llamada de LearnedCountSketch y la inicialización de buffers;
no modifica las ecuaciones de transporte, epsilon, regularización,
distribución, scheduler ni la ruta SVD. En CPU restaura los estados
transportados a CPU, y conserva `Adam.t`.

`tools/validate_moment_transport.py` prueba cuatro casos: optimizador CPU
y GPU, pesos float32 y bfloat16, estados float32; inicialización más dos
recalibraciones, forma16×24 y d8,r2. Compara M/V antes/después de init y
el siguiente paso de Adam. Referencia independiente en float64 con los
solapamientos calculados a la precisión original; atol2e-6, rtol2e-5.
Los ocho transportes pasan. Error máximo M=1.303e-7, V=1.105e-5; error
normalizado por tolerancia máximo0.01464 (<1). La preservación de los
tensores durante init es exacta. El original falla la misma prueba.

La primera versión del test rematerializaba sketches y una comparación
float32 falló incluso antes de init: las colisiones de `scatter_` pueden
alterar qué valor queda en una posición. Se conservó ese resultado local
(`moment-adapted-full-tensors`). El test definitivo captura los sketches
densos realmente devueltos dentro del transporte, sin cambiar su cálculo,
muestreo ni valores. No se ocultó el fallo ampliando tolerancias.

Comandos, logs y resultados definitivos:

```powershell
& ./LSP-OFFLOAD/tools/run_regression.ps1 -Suite moments -Source original -RunId moment-original-captured-inputs
& ./LSP-OFFLOAD/tools/run_regression.ps1 -Suite moments -Source adaptado -RunId moment-adapted-captured-inputs
& ./LSP-OFFLOAD/tools/run_regression.ps1 -Suite scheduler -Source adaptado -RunId scheduler-boundaries-before-fix
```

Cada directorio `results/<RunId>/` contiene `diagnostics.json`, `command.json`
y tensores completos `.pt`; `logs/<RunId>.log` conserva stdout/stderr.
Los IDs no se sobrescriben. El resumen versionado es
`configs/adaptation_validation.json`. Las marcas NativeCommandError de
PowerShell envuelven stderr nativo (incluido tqdm); el exit code registrado
del contenedor es0 para el parche y1 para las regresiones esperadas.

**N14 confirmado y piloto detenido.** Se ejercitó el código real de dos
capas16×16, forward/backward con pérdida finita y Adam CPU original. Zero
se activa con `Scheduler(zero=True)` (equivale a `--scheduler-zero` en el
trainer), sin compresión. LSP usa CS gaussiano d8,r2 y `zero=False`,
`fcfs_point=1`, `lcfs_d2h_delay=1`, demás delays0; es una configuración
sintética para cubrir FCFS y LCFS, no la configuración GPT2 definitiva.
Ambas rutas conservan exactamente el mismo Adam CPU con eps1e-4.

| Límite evaluado | Control serial, actualizaciones por capa | Zero | LSP FCFS/LCFS probado |
| --- | --- | --- | --- |
| Evaluación (`Trainer.train_epoch` llama clear antes de model.eval) | [1,1], pasa | [0,0], falla | [0,1], falla |
| Salida (`ScheduleEnv.__exit__` llama clear) | [1,1], pasa | [0,0], falla | [0,1], falla |

Se esperaban [1,1]. Zero tenía seis callbacks pendientes y LSP tres en la
capa0; clear los elimina. Los pesos de esas capas permanecen idénticos a
los iniciales y hay entradas sin consumir en Context.stack. Las pruebas
guardan pesos y estados completos, además de los contadores. No basta
sincronizar CUDA: esos callbacks de Python nunca fueron ejecutados.
No se interpreta la loss sintética como calidad ni se comparan sus valores
entre métodos.

**Corrección propuesta, no aplicada:** añadir un `Scheduler.drain()` que
complete operaciones pendientes en el orden de dependencias del siguiente
forward, sin efectuar otro forward ni generar gradientes nuevos; esperar
las transferencias/actualizaciones antes de evaluar, recalibrar, guardar o
terminar normalmente. `clear()` debe rechazar trabajo pendiente en esos
límites, en lugar de borrarlo. Aplicar el mismo cierre a ambas rutas y
probar conteos exactos y tensores completos W/M/V/t frente a referencia
serial con iguales gradientes/proyectores, sin duplicar actualizaciones.
Referencia: Algoritmo3 p.11, línea5 exige esperar actualización antes de
forward; en código original `Scheduler.clear:419` y `ScheduleEnv.__exit__:536`.

Recuperar actualizaciones actualmente perdidas cambia la trayectoria del
código publicado y excede el único parche matemático autorizado. Se
solicitará autorización específica antes de aplicarlo, conforme al límite
del usuario. No se descargaron modelos/datos, no hubo piloto GPT2 ni OOM,
no se activó el descenso a Medium y no se corrieron microbenchmarks de
rendimiento. No hay tiempos para estimar un presupuesto de fine-tuning.
La comparación queda pendiente de resolver este bloqueo, sin ampliar el
alcance a otras familias.

## Continuación autorizada: 14 de noviembre de 2026

Esta sección sustituye el estado de bloqueo de N14 descrito arriba; conserva
esa evidencia histórica. El segundo parche está separado del de momentos:
`adaptado/patches/0002-drain-pending-operations.patch`. Implementa drain en
evaluación, recalibración, guardado y salida normal; clear rechaza trabajo
pendiente. El orden FCFS/LCFS durante entrenamiento no cambia y las
iteraciones ordinarias no vacían la cola.

La prueba v1 detectó que aplicar hooks después del matmul usaba pesos
atrasados (Algoritmo3, p.11, línea5). Se movieron antes de consumirlos.
La v2 pasó siete casos, pero un LSP float32 mostró diferencia en W con M/V
iguales. Se protegió la vida de los buffers CUDA con record_stream en D2H
y en la actualización. No se ampliaron tolerancias ni se cambiaron fórmulas.
Los parches intermedios y resultados fallidos permanecen en sus directorios
locales. La v3 pasa los ocho casos y W/M/V coinciden exactamente con serial
en los límites observados (atol2e-6, rtol5e-5 admitidos). Los contadores y
tokens de actualización verifican aplicación exactamente una vez, incluido
doble drain, evaluación, recalibración, guardado y salida. Son doce pasos,
seis capas y las mismas entradas/pesos/proyectores en cada par. La prueba
usa d8/r1 para evitar ambigüedad de colisiones; no cambia el r8 del piloto.

Comandos definitivos (IDs ya existentes, no sobrescribir):

```powershell
& ./LSP-OFFLOAD/tools/run_regression.ps1 -Suite equivalence -Source adaptado -RunId scheduler-drain-multistep-v3
& ./LSP-OFFLOAD/tools/run_regression.ps1 -Suite gpt2 -Source adaptado -RunId scheduler-gpt2-checkpointing-v2
& ./LSP-OFFLOAD/tools/fetch_pilot_assets.ps1
& ./LSP-OFFLOAD/tools/run_comparison.ps1
```

El test GPT2 aleatorio pequeño pasa seis pasos en ambas rutas con checkpointing,
ocho matrices y embeddings congelados. El control sin hook de gradientes de
activación falla con `element 0 of tensors does not require grad and does not
have a grad_fn`. `enable_input_require_grads()` en el wrapper común corrige
la compatibilidad de checkpointing; no añade parámetros entrenables. No es
una medición de calidad de GPT2 preentrenado. El primer intento de este test
omitió dos argumentos del wrapper y falló antes de probar la propiedad;
se conserva el log y no se atribuye ese fallo al algoritmo.

Los activos reales están fijados y tienen hashes en `configs/pilot_assets.json`:
Alpaca GPT-4, 52 002 ejemplos, commit80cda626ea305004be42426671c66efebbf22144;
GPT2 Large, commit32b71b12589c2f8d625668d2335a01cac3249519. Datos y pesos
permanecen locales/excluidos. Los wrappers conservan formato, split90%,
16 ejemplos de calibración, batch4, longitud máxima512, padding múltiplo8,
bf16 y Adam CPU float32 (lr1e-4, eps1e-4) del perfil de scripts. Se mantiene
la computación original de regularización con coeficiente0. Se registra una
semilla local2026. La evaluación corta usa los primeros64 ejemplos del split
de evaluación original: perplexity sin alterar `calculate_ppl`, no un resultado
de calidad comparable a la evaluación completa del paper.

La identidad es explícita: Zero=`Scheduler(zero=True)` + compresor identidad;
LSP=`zero=False`, d512/r8, gaussian, FCFSpoint=min(32,n_modules/2), delays
FCFSprocess1/H2D2/update2 y LCFSd2h1000/process2/H2D1. Mismo Adam CPU.
LSP usa100 iteraciones comunes de ajuste,0 específicas, lr0.1 y umbral0.8
del script; las diferencias previas con el paper siguen documentadas.
El piloto acorta CheckFreq1000 a50 para atravesar recalibración real; la
comparación de rendimiento restaura1000. No se usa el nombre de un script
para deducir qué ruta se ejecuta.

**OOM real de Large:** `scheduler-v3-pilot-gpt2-large-zero-0` terminó con
exit137 y Docker `State.OOMKilled=true`, al inicializar estados del optimizador,
antes de entrenar. Wall-clock externo36.727s; última muestra RSS7 438 909 440B,
RAM disponible40 722 432B y RAM total8 171 515 904B. El proceso fue eliminado
antes de escribir resumen Python; el wrapper registra estado Docker y conserva
telemetría/log. No es un tiempo por iteración. Se activa el único descenso
autorizado a GPT2 Medium para ambos métodos, sin aumentar RAM/swap ni modificar
batch/longitud/técnica.

**Piloto Medium superado.** Checkpoint6dcaa7a952f72f9298047fd5137cd6e4f05f41da,
354 823 168 parámetros totales; ambas rutas actualizan301 989 888 parámetros
en96 matrices y congelan el resto. Cada matriz termina con contador100.
Losses finitas, pesos modificados, evaluación/cierre completados. LSP cambia
proyectores realmente en el paso50 sin incrementar contadores durante el fit.

| Diagnóstico de ejecución,100 pasos | Zero | LSP d512/r8 |
| --- | ---: | ---: |
| Tiempo total del proceso Python,s |90.457|102.157|
| Tiempo externo Docker,s (incluye arranque) |99.323|108.945|
| Entrenamiento incl. drain final,s |82.048|73.304|
| Inicialización de proyectores,s |no aplica|3.501|
| Recalibración real en paso50,s |no aplica|18.816|
| Evaluación64 ejemplos,s |1.056|1.034|
| Pico GPU allocated de PyTorch,GiB |2.146|2.177|
| Pico GPU reserved de PyTorch,GiB |3.193|3.178|
| Pico RSS muestreado,GiB |5.255|5.390|

La frecuencia50 solo prueba recalibración; estos totales no son el benchmark
principal. Las iteraciones LSP74–78 incluyen duraciones3.27–3.89s, frente a
muchas otras de aproximadamente0.4s; no se atribuye una causa sin evidencia
ni se usa esta variación como dispersión entre ejecuciones. La telemetría
completa permanece en samples.csv y los pasos en steps.csv.

| Calidad diagnóstica sobre64 ejemplos,100 pasos; no convergencia | Zero | LSP |
| --- | ---: | ---: |
| Media de loss por batch (log de perplexity original) |1.608398|1.592773|
| Perplexity original exp(media loss por batch) |4.994805|4.917368|

No son métricas de convergencia ni una comparación de calidad final. Se
conservan el tratamiento original de padding/prompt y la media por batches.
El conjunto completo de evaluación contiene5 185 ejemplos; no se confunde
con el subconjunto diagnóstico. Los IDs son
`scheduler-v3-pilot-gpt2-medium-zero-0` y `scheduler-v3-pilot-gpt2-medium-lsp-0`.
Cada ejecución guarda config, environment, command, resumen, CSV y log local;
el resumen compacto reproducible está en `configs/comparison_summary.json`,
generado por `tools/summarize_comparison.py` exclusivamente desde mediciones.

Se prepara la comparación de rendimiento con tres procesos independientes
por método, cinco pasos de calentamiento y veinte medidos; se alterna el
orden entre métodos. CheckFreq vuelve a1000, sin recalibraciones dentro de
esas ventanas. El drain posterior al calentamiento separa ventanas y el
drain final está incluido en el tiempo medido; no se drena por iteración.
La calibración inicial y evaluación se registran fuera de esa ventana.
El commit de adaptación, pruebas y wrappers precede esas seis ejecuciones.

Comprobación tras terminar los parches:99/99 archivos originales y objetos
Git coinciden, sin discrepancias de bytes ni de finales de línea. El diff
de STRONHOLD está vacío. No se modifica core.autocrlf; las advertencias de
Git sobre archivos propios no implican cambios del upstream. PDF, entornos,
modelos, datos y resultados pesados siguen excluidos.
