# PFC 1 - STRONGHOLD: dependencias de compilacion, sin alterar SHv0 original.
# Imagen de desarrollo de PyTorch 2.7.1 / CUDA 12.8 fijada por digest.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential git ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Versiones verificadas en el paso previo del proyecto.
RUN python -m pip install --no-cache-dir \
    regex==2024.11.6 \
    pybind11==2.13.6 \
    six==1.17.0 \
    numpy==1.23.5

# RTX 5050 Laptop GPU: compute capability 12.0.
# Un solo trabajo paralelo limita la RAM consumida durante la compilacion.
ENV TORCH_CUDA_ARCH_LIST=12.0 \
    MAX_JOBS=1 \
    APEX_PARALLEL_BUILD=1 \
    APEX_CPP_EXT=1 \
    APEX_CUDA_EXT=1

# NVIDIA Apex oficial, version fijada (misma referencia que usa NVIDIA Cosmos).
# Imprescindible compilar extensiones CUDA, incluido amp_C.
RUN python -m pip install -v --no-cache-dir --no-build-isolation --no-deps \
    "git+https://github.com/NVIDIA/apex@9e3568a"

# La importacion comprueba la disponibilidad de las extensiones, no entrena.
RUN python -c "import apex, amp_C; from apex.optimizers import FusedAdam; print('NVIDIA APEX IMPORTACION OK')"

WORKDIR /pfc
