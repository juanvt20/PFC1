"""PFC 1: descarga y prepara Salesforce WikiText-103 para el GPT de STRONGHOLD.

ETAPAS:
    download: baja archivos de Salesforce (revision fijada) y GPT-2/OpenAI.
    convert:  transforma Parquet en JSONL, sin modificar su texto.
    index:    ejecuta tools/preprocess_data.py ORIGINAL de Megatron/STRONGHOLD.

Los archivos se conservan en /data, un volumen de Windows/Docker.
Solo se consume la particion train para el indice Megatron. Los archivos
validation y test oficiales se conservan aparte, sin contaminarlos.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATASET_REPO = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_VARIANT = "wikitext-103-raw-v1"
DATASET_BASE = (
    f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
    f"{DATASET_REVISION}/{DATASET_VARIANT}"
)

# SHA256 de los archivos publicados por Salesforce en Hugging Face.
PARQUET_FILES = {
    "train-00000-of-00002.parquet": "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212",
    "train-00001-of-00002.parquet": "ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720",
    "validation-00000-of-00001.parquet": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
    "test-00000-of-00001.parquet": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
}

# Archivos BPE originales de GPT-2, compatibles con GPT2BPETokenizer.
GPT2_FILES = {
    "gpt2-vocab.json": "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json",
    "gpt2-merges.txt": "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path, obj):
    """Escritura atomica para no dejar manifiestos truncados."""
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as output:
        json.dump(obj, output, indent=2, ensure_ascii=False)
        output.write("\n")
    tmp.replace(path)


def download_one(url, destination, expected_sha256=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        current_sha = sha256_file(destination)
        if expected_sha256 and current_sha != expected_sha256:
            raise RuntimeError(
                f"Existe un archivo con hash diferente: {destination}. "
                "Se detiene para no reemplazarlo automaticamente."
            )
        print(f"YA DESCARGADO: {destination.name} ({current_sha})", flush=True)
        return current_sha

    tmp = destination.with_name(destination.name + ".partial")
    # Si una descarga anterior se interrumpio, se reanuda de forma segura
    # empezando de cero. No se reutilizan bytes parciales sin verificar.
    request = urllib.request.Request(url, headers={"User-Agent": "PFC1-STRONGHOLD/1.0"})
    print(f"DESCARGANDO: {destination.name}", flush=True)
    started = time.monotonic()
    bytes_downloaded = 0
    next_report = 32 * 1024 * 1024
    with urllib.request.urlopen(request, timeout=90) as response, tmp.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            bytes_downloaded += len(chunk)
            if bytes_downloaded >= next_report:
                print(
                    f"  {destination.name}: {bytes_downloaded / 1048576:.0f} MiB recibidos",
                    flush=True,
                )
                next_report += 32 * 1024 * 1024
    actual_sha = sha256_file(tmp)
    if expected_sha256 and actual_sha != expected_sha256:
        raise RuntimeError(
            f"SHA256 incorrecto en {destination.name}: "
            f"esperado={expected_sha256}, recibido={actual_sha}. "
            "Se conserva el archivo .partial para diagnostico."
        )
    if tmp.stat().st_size < 1000:
        raise RuntimeError(f"Descarga sospechosamente pequena: {tmp}")
    tmp.replace(destination)
    print(
        f"OK: {destination.name} ({destination.stat().st_size} bytes, "
        f"SHA256={actual_sha}, {time.monotonic() - started:.1f} s)",
        flush=True,
    )
    return actual_sha


def download(root):
    raw = root / "raw_parquet"
    tokenizer = root / "tokenizer"
    records = []
    for filename, expected in PARQUET_FILES.items():
        url = f"{DATASET_BASE}/{filename}"
        path = raw / filename
        actual = download_one(url, path, expected)
        records.append({"archivo": str(path), "url": url, "sha256": actual, "bytes": path.stat().st_size})

    for filename, url in GPT2_FILES.items():
        path = tokenizer / filename
        actual = download_one(url, path)
        records.append({"archivo": str(path), "url": url, "sha256": actual, "bytes": path.stat().st_size})

    vocab = json.loads((tokenizer / "gpt2-vocab.json").read_text(encoding="utf-8"))
    if not isinstance(vocab, dict) or "<|endoftext|>" not in vocab:
        raise RuntimeError("El vocabulario GPT-2 descargado no parece valido")
    with (tokenizer / "gpt2-merges.txt").open("r", encoding="utf-8") as f:
        if not f.readline().startswith("#version:"):
            raise RuntimeError("La tabla GPT-2 BPE descargada no parece valida")

    json_file(root / "01_descargas.json", {
        "etapa": "descarga_verificada",
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "variante": DATASET_VARIANT,
        "archivos": records,
    })
    print("DESCARGA WIKITEXT-103 Y GPT2: OK", flush=True)


def convert(root):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Falta pyarrow: construye Dockerfile.wikitext103") from exc

    sources = {
        "train": [root / "raw_parquet" / f"train-{i:05d}-of-00002.parquet" for i in (0, 1)],
        "validation": [root / "raw_parquet" / "validation-00000-of-00001.parquet"],
        "test": [root / "raw_parquet" / "test-00000-of-00001.parquet"],
    }
    jsonl_dir = root / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for split, paths in sources.items():
        final = jsonl_dir / f"{split}.jsonl"
        if final.exists():
            raise FileExistsError(
                f"{final} ya existe: no lo sobrescribiremos. "
                "No repitas la etapa convert si ya finalizo."
            )
        tmp = final.with_name(final.name + ".partial")
        total_rows = 0
        nonempty_rows = 0
        with tmp.open("w", encoding="utf-8", newline="\n") as writer:
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(f"Falta {path}; ejecuta primero --stage download")
                expected = PARQUET_FILES[path.name]
                if sha256_file(path) != expected:
                    raise RuntimeError(f"Hash incorrecto: {path}")
                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(batch_size=8192, columns=["text"]):
                    for value in batch.column(0).to_pylist():
                        total_rows += 1
                        if value is None or not value.strip():
                            continue
                        # No se sintetiza ni cambia el contenido textual.
                        writer.write(json.dumps({"text": value}, ensure_ascii=False) + "\n")
                        nonempty_rows += 1
                print(f"CONVERTIDO: {path.name}; filas acumuladas: {total_rows}", flush=True)
        if nonempty_rows == 0:
            raise RuntimeError(f"No hay contenido textual en la particion {split}")
        tmp.replace(final)
        manifest[split] = {
            "ruta": str(final), "filas_originales": total_rows,
            "filas_no_vacias": nonempty_rows,
            "filas_vacias_descartadas": total_rows - nonempty_rows,
            "sha256_jsonl": sha256_file(final), "bytes": final.stat().st_size,
        }
        print(f"JSONL {split}: {nonempty_rows} textos; guardado en {final}", flush=True)

    json_file(root / "02_conversion.json", {
        "etapa": "conversion_parquet_a_jsonl",
        "transformacion": "cada fila text no vacia pasa a un registro JSONL; texto sin modificar",
        "dataset_revision": DATASET_REVISION,
        "particiones": manifest,
    })
    print("CONVERSION JSONL WIKITEXT-103: OK", flush=True)


def index(root, workers):
    source = Path(__file__).resolve().parents[1]
    original_preprocessor = source / "tools" / "preprocess_data.py"
    train_jsonl = root / "jsonl" / "train.jsonl"
    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Falta {train_jsonl}; ejecuta --stage convert")
    output_prefix = root / "megatron" / "wikitext103_gpt2"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    expected_output = [
        Path(str(output_prefix) + "_text_document.bin"),
        Path(str(output_prefix) + "_text_document.idx"),
    ]
    if any(p.exists() for p in expected_output):
        raise FileExistsError(
            "Ya existe un indice Megatron en el destino. "
            "No se sobrescribira automaticamente: " + ", ".join(map(str, expected_output))
        )

    command = [
        sys.executable, "-B", str(original_preprocessor),
        "--input", str(train_jsonl),
        "--output-prefix", str(output_prefix),
        "--dataset-impl", "mmap",
        "--tokenizer-type", "GPT2BPETokenizer",
        "--vocab-file", str(root / "tokenizer" / "gpt2-vocab.json"),
        "--merge-file", str(root / "tokenizer" / "gpt2-merges.txt"),
        "--append-eod", "--workers", str(workers), "--log-interval", "100000",
    ]
    print("PREPROCESADOR ORIGINAL DE MEGATRON/STRONGHOLD:", flush=True)
    print(" ".join(command), flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=str(source), check=True)
    for path in expected_output:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"El preprocesador no genero un archivo valido: {path}")

    json_file(root / "03_indice_megatron.json", {
        "etapa": "preprocesamiento_original_megatron",
        "dataset_revision": DATASET_REVISION,
        "particion_usada": "train",
        "particiones_oficiales_validation_y_test_reservadas": True,
        "tokenizador": "GPT2BPETokenizer",
        "anade_token_eod": True,
        "formato": "mmap",
        "comando": command,
        "duracion_segundos": time.monotonic() - started,
        "archivos": [
            {"ruta": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in expected_output
        ],
        "data_path_para_entrenamiento": str(output_prefix) + "_text_document",
    })
    print(f"DATA_PATH PARA GPT: {output_prefix}_text_document", flush=True)
    print("INDICES BIN/IDX MEGATRON: OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/wikitext103")
    parser.add_argument("--stage", choices=["download", "convert", "index", "all"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers debe estar entre 1 y 8")
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage in ("download", "all"):
        download(root)
    if args.stage in ("convert", "all"):
        convert(root)
    if args.stage in ("index", "all"):
        index(root, args.workers)


if __name__ == "__main__":
    main()
