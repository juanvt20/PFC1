# STRONGHOLD — Etapa 3: primer entrenamiento real de GPT

## Antes de entrenar

El estudiante confirmó que la descarga y preparación de WikiText-103 finalizaron con:

- `DESCARGA WIKITEXT-103 Y GPT2: OK` (SHA256 de cuatro Parquet verificado).
- `CONVERSION JSONL WIKITEXT-103: OK` (1 165 029 documentos de entrenamiento no vacíos).
- `INDICES BIN/IDX MEGATRON: OK`.

El corpus no es el dataset original privado de STRONGHOLD; es la adaptación experimental pública documentada. El archivo `pfc_tools/lanzar_gpt_pfc.py` solo prepara los argumentos y ejecuta mediante `runpy` el **mismo `pretrain_gpt.py`** de STRONGHOLD. La compatibilidad de importación `torch._six.inf` se limita a DeepSpeed 0.5.8.

## Instalación

Extrae el ZIP en `C:\PFC1\STRONHOLD\pfc1-stronghold-etapa1` y comprueba:

```powershell
Test-Path .\SHv0-adaptado\pfc_tools\lanzar_gpt_pfc.py
```

Debe salir `True`.

## Primera ejecución: Megatron-LM, línea base SIN offloading

Solo 2 pasos, 4 capas, hidden size 256, 4 cabezas, longitud 128, microbatch 1, FP32. Es un **smoke test** de integración, no un benchmark ni una evaluación de calidad.

```powershell
$fecha = Get-Date -Format "yyyyMMdd_HHmmss"
docker run --rm --network none --gpus all `
  --shm-size=1g `
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29501 `
  -e RANK=0 -e WORLD_SIZE=1 -e LOCAL_RANK=0 `
  -e OMP_NUM_THREADS=4 -e MAX_JOBS=1 `
  -v "${PWD}/SHv0-adaptado:/work" `
  -v "${PWD}/pfc_data:/data" `
  -v "${PWD}/resultados:/results" `
  -w /work pfc-stronghold:ds058 `
  python -B pfc_tools/lanzar_gpt_pfc.py --mode baseline --iters 2 `
  2>&1 | Tee-Object -FilePath ".\resultados\STRONGHOLD\gpt_baseline_smoke_$fecha.log"
```

**No continúes si se produce un error:** remite todo el log para resolverlo manteniendo la lógica original. El lanzador guarda un JSON con estado, configuración, dataset, GPU, versiones y duración. Para un resultado correcto debe figurar `ENTRENAMIENTO GPT PFC1 COMPLETADO`.

## Segunda ejecución: STRONGHOLD original con offloading

**Solo después de verificar la primera ejecución**, la configuración correspondiente es:

```powershell
$fecha = Get-Date -Format "yyyyMMdd_HHmmss"
docker run --rm --network none --gpus all `
  --shm-size=1g `
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29502 `
  -e RANK=0 -e WORLD_SIZE=1 -e LOCAL_RANK=0 `
  -e OMP_NUM_THREADS=4 -e MAX_JOBS=1 `
  -v "${PWD}/SHv0-adaptado:/work" `
  -v "${PWD}/pfc_data:/data" `
  -v "${PWD}/resultados:/results" `
  -w /work pfc-stronghold:ds058 `
  python -B pfc_tools/lanzar_gpt_pfc.py --mode stronghold --iters 2 --gl-window-size 2 `
  2>&1 | Tee-Object -FilePath ".\resultados\STRONGHOLD\gpt_stronghold_smoke_$fecha.log"
```

No ejecutar ambos modos en paralelo: compiten por la RAM y la única GPU. La activación de la técnica se hace mediante `--enable-gl`, `--gl-world-size 1`, `--gl-window-size 2`, como en los ejemplos originales.

## Qué esperamos al final

En la carpeta `resultados/STRONGHOLD/` habrá un archivo log y un JSON **por ejecución**; no hay checkpoints todavía. Cuando ambos entrenen, haremos el experimento de rendimiento con duraciones mayores, tamaños de modelo más grandes y métricas (muestras/s, tiempo, RAM, VRAM), siempre con la misma configuración de datos.

**Advertencia metodológica:** este piloto de dos pasos no demuestra que STRONGHOLD acelere el entrenamiento. Sirve únicamente para probar que la implementación original puede ejecutar un ciclo completo en la RTX 5050 y que su offloading funciona. Si hay incompatibilidades PyTorch 2.7/DeepSpeed 0.5.8, se registrarán y corregirán en el árbol `SHv0-adaptado`, preservando `stronghold-original`.
