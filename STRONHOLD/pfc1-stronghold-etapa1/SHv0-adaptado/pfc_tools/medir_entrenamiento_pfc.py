"""Medicion reproducible de Megatron/STRONGHOLD sin alterar su codigo.

Lanza el adaptador existente `lanzar_gpt_pfc.py`, muestra un latido de
progreso mientras Megatron inicializa, y conserva registros y mediciones
en /results/STRONGHOLD/experimentos. No cambia el forward/backward,
el optimizador, las transferencias de STRONGHOLD ni el dataset.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "lanzar_gpt_pfc.py"

ITERATION = re.compile(
    r"\biteration\s+(\d+)\s*/\s*(\d+)\s*\|"
    r"\s*elapsed time per iteration \(ms\):\s*([\d.]+)"
)
LOSS = re.compile(r"\|\s*lm loss:\s*([\d.eE+-]+)")
LEARNING_RATE = re.compile(r"\|\s*learning rate:\s*([\d.eE+-]+)")
SPEED = re.compile(r"SamplesPerSecond:\s*([\d.eE+-]+)")
MEMORY = re.compile(r"\bmax allocated:\s*([\d.eE+-]+)")
INIT = re.compile(r"time to initialize megatron \(seconds\):\s*([\d.]+)")
SKIPPED = re.compile(r"number of skipped iterations:\s*(\d+)")
NAN = re.compile(r"number of nan iterations:\s*(\d+)")
FLOPS = re.compile(r"Effective Tera Flops per GPU:\s*([\d.]+)")
TIMERS = re.compile(r"([\w/-]+):\s*([\d.]+)")

ITERATION_COLUMNS = [
    "iteration", "total_iterations", "iteration_ms", "lm_loss",
    "learning_rate", "samples_per_second", "tflops_per_gpu",
    "skipped_iterations", "nan_iterations", "forward_ms", "backward_ms",
    "optimizer_ms", "offloading_call_ms",
]
TELEMETRY_COLUMNS = [
    "elapsed_s", "container_memory_mib", "gpu_global_memory_mib",
    "gpu_utilization_percent",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fmatch(regex, line):
    match = regex.search(line)
    return float(match.group(1)) if match else None


def parse_training_lines(lines):
    """Convierte los logs originales en filas; no estima datos ausentes."""
    rows = []
    metadata = {"initialization_s": None, "reported_gpu_peak_allocated_mib": None}
    status = {"completed_message_seen": False, "last_phase": "iniciando"}
    for line in lines:
        stripped = line.strip()
        if "compiling and loading fused kernels" in line:
            status["last_phase"] = "cargando kernels CUDA / sincronizacion"
        elif "building GPT model" in line:
            status["last_phase"] = "construyendo GPT"
        elif "building train, validation, and test datasets" in line:
            status["last_phase"] = "cargando datasets"
        elif stripped == "training ...":
            status["last_phase"] = "entrenando"
        elif "ENTRENAMIENTO GPT PFC1 COMPLETADO" in line:
            status["last_phase"] = "completado"
            status["completed_message_seen"] = True

        initialization_s = fmatch(INIT, line)
        if initialization_s is not None:
            metadata["initialization_s"] = initialization_s
        peak_mib = fmatch(MEMORY, line)
        if peak_mib is not None:
            previous = metadata["reported_gpu_peak_allocated_mib"]
            metadata["reported_gpu_peak_allocated_mib"] = max(
                peak_mib, previous if previous is not None else peak_mib
            )

        iteration_match = ITERATION.search(line)
        if iteration_match:
            number, total, elapsed = iteration_match.groups()
            row = {column: None for column in ITERATION_COLUMNS}
            row.update(
                iteration=int(number),
                total_iterations=int(total),
                iteration_ms=float(elapsed),
                lm_loss=fmatch(LOSS, line),
                learning_rate=fmatch(LEARNING_RATE, line),
            )
            skipped = SKIPPED.search(line)
            nan = NAN.search(line)
            row["skipped_iterations"] = int(skipped.group(1)) if skipped else None
            row["nan_iterations"] = int(nan.group(1)) if nan else None
            rows.append(row)
            status["last_phase"] = f"entrenando {number}/{total}"
        elif rows:
            if "SamplesPerSecond:" in line:
                rows[-1]["samples_per_second"] = fmatch(SPEED, line)
            elif "Effective Tera Flops per GPU:" in line:
                rows[-1]["tflops_per_gpu"] = fmatch(FLOPS, line)
            elif line.startswith("time (ms) |"):
                timers = dict(TIMERS.findall(line))
                for output_key, source_key in (
                    ("forward_ms", "forward-compute"),
                    ("backward_ms", "backward-compute"),
                    ("optimizer_ms", "optimizer"),
                    ("offloading_call_ms", "offloading-func-call-overhead"),
                ):
                    if source_key in timers:
                        rows[-1][output_key] = float(timers[source_key])
    return rows, metadata, status


def save_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def cgroup_memory_mib():
    try:
        return round(int(Path("/sys/fs/cgroup/memory.current").read_text()) / 1048576, 2)
    except (OSError, ValueError):
        return None


def gpu_sample():
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits", "--id=0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            timeout=5,
            check=True,
        )
        values = result.stdout.strip().split(",")
        return float(values[0].strip()), float(values[1].strip())
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None, None


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "stronghold"), required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--gl-window-size", type=int, default=4)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--monitor-seconds", type=int, default=10)
    parser.add_argument("--replica", type=int, default=1)
    parser.add_argument("--label", default="calibracion")
    parser.add_argument("--results-dir", default="/results/STRONGHOLD/experimentos")
    arguments = parser.parse_args()
    if not LAUNCHER.exists():
        parser.error(f"Falta el lanzador de etapa 3: {LAUNCHER}")
    if arguments.iters < 2 or arguments.warmup_iters < 0 or arguments.warmup_iters >= arguments.iters:
        parser.error("0 <= --warmup-iters < --iters y --iters >= 2")
    if arguments.heartbeat_seconds < 1 or arguments.monitor_seconds < 1:
        parser.error("Los intervalos de monitorizacion deben ser positivos")
    if arguments.replica < 1:
        parser.error("--replica debe ser >= 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", arguments.label):
        parser.error("--label admite solo letras, numeros, '_' y '-' (maximo 32)")
    return arguments


def summarize(rows, warmup, batch):
    effective = [r for r in rows if r["iteration"] > warmup and r["iteration_ms"] > 0]
    if not effective:
        return None
    times_ms = [r["iteration_ms"] for r in effective]
    total_seconds = sum(times_ms) / 1000
    mean_ms = sum(times_ms) / len(times_ms)
    return {
        "measured_iterations": len(effective),
        "warmup_iterations_excluded": warmup,
        "mean_iteration_ms": round(mean_ms, 5),
        "throughput_samples_per_second": round(len(effective) * batch / total_seconds, 6),
        "min_iteration_ms": min(times_ms),
        "max_iteration_ms": max(times_ms),
        "mean_lm_loss_observed": round(
            sum(r["lm_loss"] for r in effective if r["lm_loss"] is not None)
            / sum(r["lm_loss"] is not None for r in effective), 7
        ) if any(r["lm_loss"] is not None for r in effective) else None,
    }


def main():
    args = options()
    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_id = (f"{args.label}_{args.mode}_L{args.layers}_H{args.hidden_size}"
              f"_S{args.seq_length}_B{args.batch_size}_N{args.iters}_R{args.replica}_{timestamp}")
    log_file = result_dir / f"{run_id}.log"
    csv_file = result_dir / f"{run_id}_iteraciones.csv"
    telemetry_file = result_dir / f"{run_id}_telemetria.csv"
    summary_file = result_dir / f"{run_id}_resumen.json"

    command = [
        sys.executable, "-u", "-B", str(LAUNCHER),
        "--mode", args.mode, "--layers", str(args.layers),
        "--hidden-size", str(args.hidden_size), "--heads", str(args.heads),
        "--seq-length", str(args.seq_length), "--batch-size", str(args.batch_size),
        "--iters", str(args.iters), "--gl-window-size", str(args.gl_window_size),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(HERE.parent) + os.pathsep + environment.get("PYTHONPATH", "")
    began = time.monotonic()
    started_utc = utc_now()
    messages = []
    telemetry = []
    running = {"last_output": began, "phase": "arrancando PyTorch", "last_iteration": 0}
    run_error = None
    exit_code = None

    print(f"[PFC] Iniciando {run_id}", flush=True)
    print(f"[PFC] Log: {log_file}", flush=True)
    print(f"[PFC] Se informara actividad cada {args.heartbeat_seconds} segundos.", flush=True)
    with log_file.open("w", encoding="utf-8", buffering=1) as output:
        output.write(f"[PFC] Comando: {' '.join(command)}\n")
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, encoding="utf-8", errors="replace", env=environment,
            cwd=str(HERE.parent),
        )

        def copy_output():
            assert proc.stdout is not None
            for line in proc.stdout:
                output.write(line)
                print(line, end="", flush=True)
                messages.append(line)
                running["last_output"] = time.monotonic()
                line_rows, _, parsed_status = parse_training_lines((line,))
                if line_rows:
                    running["last_iteration"] = line_rows[-1]["iteration"]
                    pct = 100 * running["last_iteration"] / args.iters
                    print(f"[PFC] PROGRESO: {running['last_iteration']}/{args.iters} ({pct:.1f}%)", flush=True)
                if "compiling and loading fused kernels" in line:
                    running["phase"] = "cargando kernels / sincronizando NCCL"
                elif "time to initialize megatron" in line:
                    running["phase"] = "inicializacion completada"
                elif "building GPT model" in line:
                    running["phase"] = "construyendo modelo"
                elif "building train, validation, and test datasets" in line:
                    running["phase"] = "preparando dataset"
                elif line.strip() == "training ...":
                    running["phase"] = "entrenando"

        reader = threading.Thread(target=copy_output, name="read_megatron_stdout", daemon=True)
        reader.start()
        next_heartbeat = began + args.heartbeat_seconds
        next_monitor = began
        try:
            while proc.poll() is None:
                now = time.monotonic()
                if now >= next_monitor:
                    mem_gpu, util_gpu = gpu_sample()
                    telemetry.append({
                        "elapsed_s": round(now - began, 2),
                        "container_memory_mib": cgroup_memory_mib(),
                        "gpu_global_memory_mib": mem_gpu,
                        "gpu_utilization_percent": util_gpu,
                    })
                    next_monitor = now + args.monitor_seconds
                if now >= next_heartbeat:
                    waiting = int(now - running["last_output"])
                    print(
                        f"[PFC] Sigue activo ({int(now - began)} s), "
                        f"fase: {running['phase']}; "
                        f"sin mensajes nuevos: {waiting} s",
                        flush=True,
                    )
                    next_heartbeat = now + args.heartbeat_seconds
                time.sleep(0.5)
            exit_code = proc.wait()
        except KeyboardInterrupt:
            run_error = "Interrumpido por usuario (Ctrl+C)"
            print("[PFC] Interrupcion solicitada; terminando proceso hijo...", flush=True)
            proc.terminate()
            try:
                exit_code = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                exit_code = proc.wait()
        finally:
            reader.join(timeout=10)
            if proc.stdout:
                proc.stdout.close()

    rows, meta, state = parse_training_lines(messages)
    save_csv(csv_file, rows, ITERATION_COLUMNS)
    save_csv(telemetry_file, telemetry, TELEMETRY_COLUMNS)
    result = summarize(rows, args.warmup_iters, args.batch_size)
    complete = (run_error is None and exit_code == 0
                and state["completed_message_seen"]
                and rows and rows[-1]["iteration"] == args.iters)
    payload = {
        "run_id": run_id,
        "estado": "completado" if complete else "interrumpido" if run_error else "error",
        "error": run_error,
        "exit_code": exit_code,
        "start_utc": started_utc,
        "end_utc": utc_now(),
        "wall_time_seconds": round(time.monotonic() - began, 3),
        "configuration": vars(args),
        "megatron_initialization_seconds": meta["initialization_s"],
        "gpu_peak_allocated_mib_reported_by_megatron": meta["reported_gpu_peak_allocated_mib"],
        "gpu_global_memory_mib_peak_observed": max(
            (t["gpu_global_memory_mib"] for t in telemetry
             if t["gpu_global_memory_mib"] is not None), default=None
        ),
        "container_memory_mib_peak_observed": max(
            (t["container_memory_mib"] for t in telemetry
             if t["container_memory_mib"] is not None), default=None
        ),
        "observed_iterations": len(rows),
        "steady_state_metrics": result,
        "warning": (
            "La memoria global reportada por nvidia-smi incluye otros procesos. "
            "La exclusion de iteraciones de calentamiento es una decision de medicion. "
            "No concluir convergencia o precision del modelo a partir de este benchmark."
        ),
        "files": {"log": log_file.name, "iterations_csv": csv_file.name,
                  "telemetry_csv": telemetry_file.name},
    }
    with summary_file.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(f"[PFC] Resultado: {payload['estado']}", flush=True)
    print(f"[PFC] Resumen: {summary_file}", flush=True)
    if result:
        print(f"[PFC] Tiempo medio post-calentamiento: {result['mean_iteration_ms']} ms", flush=True)
        print(f"[PFC] Throughput post-calentamiento: {result['throughput_samples_per_second']} muestras/s", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
