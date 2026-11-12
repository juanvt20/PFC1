"""PFC 1: prueba numerica aislada del CPU Adam original de STRONGHOLD.

No entrena un modelo, no modifica el algoritmo y no mide rendimiento.
Compara tres actualizaciones AdamW sobre el mismo parametro y gradientes.
Guarda SIEMPRE los resultados en un JSON fechado del volumen de salida.
"""

import argparse
import json
import math
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
os.chdir(SOURCE_ROOT)
os.environ.setdefault("MAX_JOBS", "1")

# Compatibilidad de un import heredado; no cambia los calculos del algoritmo.
if "torch._six" not in sys.modules:
    old_six = types.ModuleType("torch._six")
    old_six.inf = math.inf
    sys.modules["torch._six"] = old_six

import torch  # noqa: E402
from megatron.optimizer import GL_DeepSpeedCPUAdam, load_utils  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/results/STRONGHOLD")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no es visible; comprueba Docker --gpus all")

    print("=== Validacion aislada CPU Adam STRONGHOLD (sin entrenamiento) ===", flush=True)
    print("Compilando/cargando extensiones del optimizador...", flush=True)
    load_utils(SimpleNamespace(rank=0))

    torch.manual_seed(1234)
    base = torch.randn(256, dtype=torch.float32, device="cpu")
    param_stronghold = torch.nn.Parameter(base.clone())
    param_pytorch = torch.nn.Parameter(base.clone())

    options = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
    opt_stronghold = GL_DeepSpeedCPUAdam(
        [param_stronghold], **options, adamw_mode=True, fp32_optimizer_states=True
    )
    opt_pytorch = torch.optim.AdamW([param_pytorch], **options, foreach=False)

    rows = []
    for step in range(1, 4):
        grad = torch.randn_like(base)
        param_stronghold.grad = grad.clone()
        param_pytorch.grad = grad.clone()

        opt_stronghold.step(0, 1)  # indices de grupos usados por STRONGHOLD
        opt_pytorch.step()

        state_s = opt_stronghold.state[param_stronghold]
        state_p = opt_pytorch.state[param_pytorch]
        row = {
            "step": step,
            "params_error_abs_max": float((param_stronghold-param_pytorch).abs().max()),
            "exp_avg_error_abs_max": float((state_s["exp_avg"]-state_p["exp_avg"]).abs().max()),
            "exp_avg_sq_error_abs_max": float((state_s["exp_avg_sq"]-state_p["exp_avg_sq"]).abs().max()),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = output_dir / f"validacion_cpuadam_{timestamp}.json"
    metadata = {
        "tipo": "validacion_numerica_aislada_sin_entrenamiento",
        "comparacion": "STRONGHOLD GL_DeepSpeedCPUAdam (adamw_mode=True) vs torch.optim.AdamW",
        "semilla": 1234,
        "dtype": "fp32",
        "parametros": 256,
        "pasos": 3,
        "configuracion": {"lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "valores": rows,
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RESULTADOS GUARDADOS: {path}", flush=True)
    print("VALIDACION CPU ADAM FINALIZADA (revisar diferencias reportadas)", flush=True)


if __name__ == "__main__":
    main()
