# Entorno independiente de LSP-Offload

Imagen local: `pfc-lsp-offload:torch271-cu128`, construida con el Dockerfile de esta carpeta. No se modificaron imágenes de STRONHOLD ni de otros proyectos. La compilación usa solo las cuatro fuentes originales del kernel, sin sus binarios CPython3.10.

Base: PyTorch2.7.1/CUDA12.8/cuDNN9 de desarrollo, fijada por digest. Python3.11.13, nvcc12.8.61 y g++11.4.0. Las versiones de `requirements.txt` son un perfil de compatibilidad local, no versiones originales atribuidas a los autores. La primera resolución trajo PyArrow26 incompatible con NumPy1.26; la prueba de importación permitió detectarlo y fijar PyArrow17.0.0. La construcción final usa `constraints.txt` con las 154 versiones resueltas.

Se compiló `python setup.py build_ext --inplace` con `TORCH_CUDA_ARCH_LIST=12.0` y `MAX_JOBS=1`. El build puede hacerse sin GPU. Los kernels CPU requieren float32, AVX2 y tamaños múltiplo de8. Las pruebas pequeñas CPU/GPU pasan contra la fórmula específica de cada ruta; no prueban equivalencia numérica entre ellas. N15 (reinicio de momentos) sigue bloqueando la aceptación científica.

No se instalaron drivers Linux ni se modificaron paquetes globales, `.wslconfig`, RAM o swap. No se ejecutaron scripts completos de entrenamiento ni se descargaron modelos/datasets.

## Comandos y registros

Desde `C:\PFC1`:

```powershell
$dockerExe = 'C:\Users\juan\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe'
& $dockerExe build --pull=false --progress plain -t pfc-lsp-offload:torch271-cu128 -f LSP-OFFLOAD/docker/Dockerfile LSP-OFFLOAD
& $dockerExe run --rm --pull=never --network none --gpus all `
  --mount 'type=bind,source=C:\PFC1\LSP-OFFLOAD,target=/work' `
  --mount 'type=bind,source=C:\PFC1\LSP-OFFLOAD\original\upstream,target=/work/original/upstream,readonly' `
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e OMP_NUM_THREADS=2 `
  pfc-lsp-offload:torch271-cu128 python tools/validate_environment.py --run-id environment-validation-locked
```

El script rechaza reutilizar un directorio de resultados existente. El
umbral -1 fuerza el segundo fit exclusivamente para diagnóstico sintético.
Un exit code0 indica que las comprobaciones de entorno/kernel se ejecutaron,
no que se acepte el algoritmo. El venv de PDF es independiente de la imagen.

- `configs/image_identity.txt`: identidad inmutable local; no implica publicación en un registro.
- `configs/environment_packages.json`: inventario nombre/versión.
- `docker/constraints.txt`: versiones resueltas usadas por el build final; no contiene hashes de ruedas.
- `configs/environment_validation.json`: resumen del último diagnóstico y `algorithm_accepted=false`.
- `logs/build-environment.log`, `build-environment-final.log`, `build-environment-locked.log`: build inicial, corrección PyArrow y build con restricciones. Los fallos iniciales también se conservan.
- `results/*/environment.freeze.txt`: freeze bruto de inventario. Sus rutas conda `file://` no son requisitos instalables fuera de la base.

`.dockerignore` excluye PDF, venv, modelos y resultados del contexto. Los
binarios son derivados locales en la imagen; no se añaden a Git. Consultar
VALIDACION_LOCAL.md y CONTRASTE_PDF_CODIGO.md antes de continuar con calidad
o rendimiento: no se midieron throughput, convergencia ni planificador completo.
