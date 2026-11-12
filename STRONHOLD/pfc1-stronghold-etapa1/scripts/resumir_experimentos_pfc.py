"""Agrega en CSV las mediciones de STRONGHOLD y su baseline Megatron.

Solo usa datos observados: ignora corridas incompletas, evita mezclar
configuraciones distintas y calcula media geometrica de throughput.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "label", "layers", "hidden_size", "heads", "seq_length", "batch_size",
    "iters", "warmup_iters", "method", "gl_window_size", "runs",
    "geomean_samples_per_s", "mean_iteration_ms", "speedup_vs_baseline",
    "max_gpu_allocated_mib_reported", "max_container_memory_mib_observed",
]


def clean_number(value, decimals=4):
    return round(value, decimals) if value is not None else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/results/STRONGHOLD/experimentos")
    parser.add_argument("--out", default="/results/STRONGHOLD/experimentos/resumen_comparativo.csv")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"No existe la carpeta: {input_dir}")

    grouped = defaultdict(list)
    excluded = []
    for path in sorted(input_dir.glob("*_resumen.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("estado") != "completado" or not data.get("steady_state_metrics"):
                excluded.append((path.name, "corrida incompleta"))
                continue
            c = data["configuration"]
            m = data["steady_state_metrics"]
            speed = m["throughput_samples_per_second"]
            if speed <= 0:
                raise ValueError("throughput no positivo")
            group = (
                c["label"], c["layers"], c["hidden_size"], c["heads"],
                c["seq_length"], c["batch_size"], c["iters"], c["warmup_iters"],
                c["mode"], c["gl_window_size"] if c["mode"] == "stronghold" else "",
            )
            grouped[group].append(data)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            excluded.append((path.name, str(exc)))

    base_rates = {}
    for key, runs in grouped.items():
        if key[8] == "baseline":
            base_rates[key[:8]] = math.exp(sum(
                math.log(d["steady_state_metrics"]["throughput_samples_per_second"])
                for d in runs
            ) / len(runs))

    table = []
    for key, runs in sorted(grouped.items()):
        rates = [d["steady_state_metrics"]["throughput_samples_per_second"] for d in runs]
        mean_geo = math.exp(sum(math.log(x) for x in rates) / len(rates))
        durations = [d["steady_state_metrics"]["mean_iteration_ms"] for d in runs]
        gpu_peak = [d["gpu_peak_allocated_mib_reported_by_megatron"] for d in runs
                    if d.get("gpu_peak_allocated_mib_reported_by_megatron") is not None]
        ram_peak = [d["container_memory_mib_peak_observed"] for d in runs
                    if d.get("container_memory_mib_peak_observed") is not None]
        baseline = base_rates.get(key[:8])
        speedup = mean_geo / baseline if baseline and key[8] == "stronghold" else None
        table.append(dict(zip(FIELDS, [
            *key[:8], key[8], key[9], len(runs), clean_number(mean_geo),
            clean_number(sum(durations) / len(durations)), clean_number(speedup),
            clean_number(max(gpu_peak) if gpu_peak else None),
            clean_number(max(ram_peak) if ram_peak else None),
        ])))

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)
    print(f"Corridas completas leidas: {sum(map(len, grouped.values()))}")
    print(f"Corridas excluidas: {len(excluded)}")
    for filename, reason in excluded:
        print(f"  EXCLUIDA: {filename} ({reason})")
    print(f"Grupos comparables: {len(table)}")
    for row in table:
        print(
            f"{row['method']:10s} L{row['layers']} H{row['hidden_size']} "
            f"S{row['seq_length']} B{row['batch_size']} "
            f"repeticiones={row['runs']} throughput={row['geomean_samples_per_s']} "
            f"speedup={row['speedup_vs_baseline']}"
        )
    print(f"CSV comparativo: {output}")


if __name__ == "__main__":
    main()
