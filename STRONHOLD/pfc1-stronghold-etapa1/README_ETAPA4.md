# PFC1 - STRONGHOLD: Etapa 4, mediciones iniciales reproducibles

Esta etapa **no modifica** `stronghold-original/`, `pretrain_gpt.py`, ni
el algoritmo original de STRONGHOLD. Agrega dos archivos independientes:

1. `SHv0-adaptado/pfc_tools/medir_entrenamiento_pfc.py`: ejecuta el lanzador
   ya validado de etapa 3, informa actividad cada 30 s, registra progreso
   por iteracion y conserva logs, CSV, JSON y telemetria.
2. `scripts/resumir_experimentos_pfc.py`: agrupa repeticiones comparables,
   calcula media geometrica de muestras/s y speedup frente a Megatron-LM.

## Paso 0: colocar archivos (sin sobrescribir originales)

Estando en `C:\PFC1\STRONHOLD\pfc1-stronghold-etapa1` en PowerShell:

```powershell
Expand-Archive -Path .\pfc1_STRONGHOLD_experimentos_etapa4.zip -DestinationPath .
Test-Path .\SHv0-adaptado\pfc_tools\medir_entrenamiento_pfc.py
Test-Path .\scripts\resumir_experimentos_pfc.py
```

Ambos `Test-Path` deben dar `True`. No uses `-Force`: esta entrega solo agrega archivos.

## Paso 1: primera calibracion real, baseline (200 iteraciones)

Manteniendo exactamente los mismos parametros del modelo y dataset para
ambos metodos. Esta configuracion se propone para tu RTX 5050 de 8 GiB;
**todavia no esta validada en tu laptop**.

```powershell
docker run --rm --network none --gpus all `
  --shm-size=1g `
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29501 `
  -e RANK=0 -e WORLD_SIZE=1 -e LOCAL_RANK=0 `
  -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1 `
  -e OMP_NUM_THREADS=4 -e MAX_JOBS=1 `
  -v "${PWD}/SHv0-adaptado:/work" `
  -v "${PWD}/pfc_data:/data" `
  -v "${PWD}/resultados:/results" `
  -w /work pfc-stronghold:ds058 `
  python -u -B pfc_tools/medir_entrenamiento_pfc.py `
  --mode baseline --layers 8 --hidden-size 512 --heads 8 `
  --seq-length 256 --batch-size 2 --iters 200 `
  --warmup-iters 20 --label calibracion --replica 1
```

**No ejecutes los dos entrenamientos en paralelo**: compiten por GPU, CPU y RAM.
Confirma la terminacion del baseline antes de ejecutar STRONGHOLD.

## Paso 2: misma calibracion con STRONGHOLD

Solo cambian el metodo y la ventana; se mantiene el resto identico:

```powershell
docker run --rm --network none --gpus all `
  --shm-size=1g `
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29501 `
  -e RANK=0 -e WORLD_SIZE=1 -e LOCAL_RANK=0 `
  -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1 `
  -e OMP_NUM_THREADS=4 -e MAX_JOBS=1 `
  -v "${PWD}/SHv0-adaptado:/work" `
  -v "${PWD}/pfc_data:/data" `
  -v "${PWD}/resultados:/results" `
  -w /work pfc-stronghold:ds058 `
  python -u -B pfc_tools/medir_entrenamiento_pfc.py `
  --mode stronghold --layers 8 --hidden-size 512 --heads 8 `
  --seq-length 256 --batch-size 2 --iters 200 --gl-window-size 4 `
  --warmup-iters 20 --label calibracion --replica 1
```

## Paso 3: obtener tabla comparativa

Una vez completadas ambas corridas:

```powershell
docker run --rm --network none `
  -v "${PWD}/scripts:/scripts:ro" `
  -v "${PWD}/resultados:/results" `
  pfc-stronghold:ds058 `
  python -B /scripts/resumir_experimentos_pfc.py
```

La tabla estara en:

`resultados\STRONGHOLD\experimentos\resumen_comparativo.csv`

Tambien se generan cuatro archivos por ejecucion:

- `... .log`: salida original y errores, para trazabilidad.
- `... _iteraciones.csv`: tiempo, perdida, throughput y desglose de tiempos.
- `... _telemetria.csv`: uso global GPU y memoria de contenedor con muestreo.
- `... _resumen.json`: parametros, estado, tiempos y medias post-calentamiento.

**Importante:** la memoria global de `nvidia-smi` incluye otros procesos.
`max allocated` de Megatron mide memoria asignada por PyTorch y no es
equivalente al total de VRAM. Ninguna cifra ausente se inventa.

## Como interpreta el progreso

- Durante inicializacion: cada 30 s `[PFC] Sigue activo ...` indica que el
  proceso esta vivo, pero no garantiza avance en esa fase.
- Durante entrenamiento: cada iteracion completada reporta
  `[PFC] PROGRESO: 100/200 (50.0%)` y la perdida del log original.
- Si ocurre un error, tambien se guardan los archivos parciales con
  `estado=error`; no se consideran resultados experimentales completos.

## Importante antes de un entrenamiento de horas

- Este primer benchmark de rendimiento **no guarda pesos/checkpoints**:
  el lanzador de etapa 3 no los habilita. No equivale a entrenar durante una
  epoca ni a obtener un modelo convergente. La siguiente etapa agregara
  checkpoints y reanudacion para entrenamientos extensos sin alterar Megatron.
- No compares la `lm loss` de las dos corridas como una medida de calidad
  hasta verificar igualdad de inicializacion y validacion reproducible.
- El `--data-path` actual divide el conjunto `train` de WikiText-103 con
  `--split 949,50,1`; **no usa** las particiones oficiales de test/validacion.
  Para evaluaciones de calidad debemos preparar y documentar el uso de esas
  particiones antes de publicar resultados finales.
- En el paper los autores realizan **10 repeticiones por caso** y usan media
  geometrica; aqui empezamos con una sola replica de calibracion. Tras
  verificar recursos, repetiremos el experimento y escalaremos el modelo.
- Separamos tiempo de inicializacion y throughput post-calentamiento. El
  paper incluye el costo de perfilado del metodo en sus mediciones globales,
  por lo que ambos resultados deben etiquetarse correctamente.
- La comparacion de esta etapa abarca solo Megatron-LM vs STRONGHOLD. Los
  otros baselines del paper (L2L, ZeRO-Offload, ZeRO-Infinity) aun no se han
  ejecutado y no deben presentarse como comparados.

## Justificacion de la configuracion inicial

L=8, H=512, heads=8, seq=256, batch=2, fp32 se escogieron como un punto
de calibracion conservador para la RTX 5050 de 8 GiB y Docker/WSL de unos
7.6 GiB. **No son los parametros originales de los experimentos a escala
de miles de millones del paper**. El objetivo es calibrar la memoria y
tiempo para seleccionar despues modelos mas exigentes, y documentar toda
desviacion por hardware.

No borres `resultados` ni `pfc_data`: el primer directorio contiene
evidencias y el segundo dataset e indices para reutilizar.
