# STRONGHOLD: etapa de compilacion en RTX 5050

Fuente: repositorio original `xshaun/sc22-ae`, commit
`229071a8919a6918323824da93c9011f5fb1bd84`, subdirectorio `SHv0`.

## Objetivo y estado

Esta es una **copia adaptada de trabajo**, no una nueva tecnica de entrenamiento.
Su unico objetivo actual es investigar la compilacion de las extensiones C++/CUDA
originales bajo PyTorch 2.7.1, CUDA 12.8 y GPU NVIDIA RTX 5050 Laptop (`sm_120`).
No se han ejecutado entrenamientos ni obtenido resultados experimentales.

Se debe mantener `stronghold-original/SHv0` sin cambios para poder comparar
cualquier adaptacion con la fuente original.

## Cambios iniciales de compatibilidad

1. `megatron/optimizer/__init__.py`, funcion `load_utils`: se sustituye la
   arquitectura de compilacion antigua por `compute_120/sm_120`; se eliminan
   opciones que referencian Python 3.8 y Boost.Python para Python 3.8, y se
   colocan las opciones de enlace OpenMP/CURAND en `extra_ldflags`. Se utiliza
   C++17 en lugar del C++14 especificado por el script para Linux antiguo.
2. `megatron/fused_kernels/__init__.py`: arquitectura de compilacion `sm_120`
   y compilacion C++17, sin alterar la implementacion de los kernels.
3. `pfc_tools/compile_optimizer.py`: nuevo diagnostico externo; crea en memoria
   una compatibilidad para `torch._six.inf` (igual a `math.inf`) y llama a
   `megatron.optimizer.load_utils()` para **compilar e importar** los tres
   modulos del repositorio. No hace ningun entrenamiento.

No se alteraron intencionadamente el calendario de intercambio CPU/GPU,
la actualizacion de parametros del optimizador, las capas Transformer,
la perdida ni la definicion del modelo.

## Prueba segura en PowerShell

Tras descomprimir esta carpeta `SHv0-adaptado` dentro de
`C:\PFC1\STRONHOLD\pfc1-stronghold-etapa1`, ejecutar desde esa ubicacion:

```powershell
docker run --rm --network none --gpus all -v "${PWD}/SHv0-adaptado:/work" -w /work pfc-stronghold:ds058 python -B pfc_tools/compile_optimizer.py
```

Esta prueba escribe solo archivos de compilacion dentro de la copia
`SHv0-adaptado/megatron/optimizer/build/`; la carpeta fuente original no se
monta dentro del contenedor. No ejecutar los scripts originales de `examples/`
para esta fase: varios contienen rutas antiguas y operaciones de borrado.

Si falla, registrar la salida completa o al menos el primer error del
compilador y las lineas siguientes. **La compilacion aun no ha sido validada**:
las APIs C++ de PyTorch han cambiado desde el artefacto de 2022 y podria ser
necesario otro ajuste de compatibilidad. Cualquier ajuste posterior debera
documentarse antes de interpretar resultados numericos.
