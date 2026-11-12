# PFC 1 · STRONGHOLD · Etapa 1: código original y diagnóstico

## Estado actual

Esta entrega contiene una copia **sin modificaciones** del código de evaluación
publicado por los autores de STRONGHOLD y una prueba de diagnóstico para el
equipo del proyecto. **Todavía no es una adaptación funcional de STRONGHOLD a
PyTorch 2.7.1** ni constituye un resultado experimental.

La reproducción del algoritmo comenzará tras verificar y adaptar sus
dependencias de manera controlada. No se han reemplazado ni simplificado
los mecanismos originales de offloading.

## Procedencia reproducible

- Paper: *STRONGHOLD: Fast and Affordable Billion-Scale Deep Learning Model Training* (SC22).
- Código de evaluación: https://github.com/xshaun/sc22-ae
- Commit guardado: `229071a8919a6918323824da93c9011f5fb1bd84`.
- Directorio: `stronghold-original/`.
- Entorno original reportado en el repositorio: Ubuntu 20.04, CUDA 11.6,
  PyTorch 1.10.2 y una GPU V100 de 32 GB.

## Equipo objetivo

- Lenovo LOQ, Intel i7-13700HX, NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM).
- RAM física actual: 16 GB; posible ampliación futura a 32 GB.
- Windows 11 + WSL2 Ubuntu 22.04 + Docker Desktop.
- Contenedor de diagnóstico: PyTorch 2.7.1, CUDA 12.8.

## Prueba segura de esta etapa

1. Extrae el ZIP en una carpeta exclusiva para PFC 1, por ejemplo
   `C:\PFC1\STRONGHOLD`.
2. Abre **PowerShell** en la carpeta donde ves `README.md` y `scripts`.
3. Inicia Docker Desktop.
4. Ejecuta:

   ```powershell
   docker run --rm --gpus all -v "${PWD}:/pfc" -w /pfc pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime python scripts/preflight.py
   ```

La prueba valida que el contenedor ve los archivos oficiales, usa la GPU,
reconoce su capacidad de cómputo y realiza una multiplicación de matrices.
No importa ni ejecuta el código de entrenamiento antiguo, no instala paquetes
y no modifica archivos del host.

## Precaución con los scripts originales

**NO ejecutes directamente** `stronghold-original/examples/run.sh` ni
`stronghold-original/SHv0/examples/sc22-gpt-sh.sh` en tu Ubuntu o Python
habituales. El segundo lanzador contiene operaciones que:

- borran archivos de `/tmp` y checkpoints del directorio actual;
- copian modificaciones sobre archivos instalados de PyTorch y DeepSpeed;
- requieren rutas de datasets que existían en el entorno de evaluación de los autores.

Antes de entrenar se aislarán esas operaciones en un contenedor apropiado,
se revisarán los parches y se preparará una ejecución que no afecte otros
proyectos.

## Riesgos de compatibilidad identificados

1. PyTorch 1.10.2 / CUDA 11.6 no proporciona soporte moderno para la RTX 5050
   (Blackwell, capacidad de cómputo 12.0).
2. STRONGHOLD integra modificaciones en componentes internos de
   `torch.distributed` y DeepSpeed; no es suficiente cambiar la versión de
   PyTorch para afirmar que el método ya funciona.
3. La versión original de Megatron-LM y algunas extensiones C++/CUDA necesitan
   ser comprobadas contra PyTorch 2.7.1.
4. Las rutas y los datos de entrenamiento de los autores requieren
   preparación independiente.
5. La capacidad de RAM disponible dentro de WSL2 debe documentarse en cada
   experimento. Un SSD adicional no equivale a RAM.

## Próximos hitos

1. Verificar la ejecución de PyTorch en el contenedor con la prueba incluida.
2. Aislar las modificaciones de STRONGHOLD y resolver incompatibilidades de
   PyTorch/DeepSpeed/CUDA, conservando el comportamiento del algoritmo.
3. Preparar las configuraciones GPT-like y baselines contempladas en el
   repositorio original.
4. Validar numéricamente el entrenamiento y recoger medidas **reales** de
   memoria, throughput, tiempo por iteración y ventana de trabajo.
5. Registrar las versiones y los cambios respecto a los autores para la
   sección de experimentos del PFC 1.

No se deben presentar números de los autores como resultados obtenidos
en la laptop.
