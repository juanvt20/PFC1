"""Diagnóstico de solo lectura del entorno para PFC 1 / STRONGHOLD.

No importa ni ejecuta el código STRONGHOLD 2022. Solo prueba PyTorch/CUDA,
revisa la presencia de archivos y muestra memoria disponible.
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED = (
    "stronghold-original/README.md",
    "stronghold-original/examples/run.sh",
    "stronghold-original/SHv0/examples/sc22-gpt-sh.sh",
    "stronghold-original/SHv0/megatron/training.py",
    "stronghold-original/Megatron-LM/pretrain_gpt.py",
    "stronghold-original/DeepSpeedExample",
    "stronghold-original/L2L",
)


def memory_gib() -> float | None:
    meminfo = pathlib.Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / (1024 * 1024), 2)
    return None


def main() -> int:
    missing = [rel for rel in EXPECTED if not (ROOT / rel).exists()]
    print("=== Fuente original STRONGHOLD ===")
    print("Commit: 229071a8919a6918323824da93c9011f5fb1bd84")
    print("Archivos requeridos:", "OK" if not missing else "FALTANTES")
    for rel in missing:
        print(" -", rel)

    print("\n=== Entorno de ejecución ===")
    print("Python:", sys.version.split()[0])
    print("Sistema:", platform.platform())
    print("RAM visible en contenedor (GiB):", memory_gib())

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print("ERROR al importar PyTorch:", repr(exc))
        return 1

    print("PyTorch:", torch.__version__)
    print("CUDA con la que se compiló PyTorch:", torch.version.cuda)
    if not torch.cuda.is_available():
        print("ERROR: PyTorch no detecta la GPU CUDA")
        return 1

    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name)
    print("Capacidad de cómputo:", torch.cuda.get_device_capability(0))
    print("VRAM física (GiB):", round(props.total_memory / 1024**3, 2))

    # Comprueba un kernel CUDA real, no únicamente la detección de la GPU.
    x = torch.ones((2, 2), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    assert torch.all(y == 2), "Resultado CUDA inesperado"
    print("Operación CUDA:", json.dumps(y.cpu().tolist()))

    print("\nDIAGNÓSTICO BÁSICO CORRECTO")
    print("STRONGHOLD AÚN NO SE HA EJECUTADO")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
