"""PFC 1: diagnostico de compilacion del optimizador original STRONGHOLD.

No entrena, no descarga datos y no ejecuta los scripts de ejemplos.
La fuente SHv0 original debe conservarse por separado.
"""

import math
import os
import sys
import types
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
os.chdir(SOURCE_ROOT)

# Puente exclusivo de compatibilidad con PyTorch >= 2: DeepSpeed 0.5.8 y el
# Megatron-LM del artefacto utilizan solo "inf" desde "torch._six".
if "torch._six" not in sys.modules:
    torch_six = types.ModuleType("torch._six")
    torch_six.inf = math.inf
    sys.modules["torch._six"] = torch_six

import torch  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from megatron.optimizer import load_utils  # noqa: E402


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No hay GPU CUDA visible en el contenedor")

    device = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print("=== Compilacion de extensiones originales STRONGHOLD ===", flush=True)
    print(f"GPU: {device}; compute capability: {capability}", flush=True)
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}", flush=True)
    print(f"MAX_JOBS: {os.environ.get('MAX_JOBS', 'no fijado')}", flush=True)
    print("No se entrenara ningun modelo.", flush=True)

    if capability != (12, 0):
        raise RuntimeError(
            "Esta copia esta configurada unicamente para GPU sm_120; "
            "no se iniciara la compilacion en otra arquitectura."
        )

    # Compila y carga los tres modulos del proyecto, sin invocar un optimizador.
    load_utils(SimpleNamespace(rank=0))
    print("COMPILACION DE OPTIMIZADOR STRONGHOLD: OK", flush=True)


if __name__ == "__main__":
    main()
