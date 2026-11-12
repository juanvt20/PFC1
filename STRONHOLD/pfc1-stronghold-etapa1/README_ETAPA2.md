# PFC 1 — STRONGHOLD: WikiText-103, etapa de datos

Esta etapa prepara un **dataset público real** para el **GPT del código original de STRONGHOLD**. No contiene un modelo reimplementado, datos inventados ni resultados de entrenamiento.

**Importante para la metodología:** el artículo STRONGHOLD no publica un corpus específico y sus ejemplos hacen referencia a archivos locales privados. WikiText-103 *no se debe presentar como el dataset original del paper*. Es una adaptación reproducible elegida para los experimentos en la RTX 5050.

## Qué contiene el ZIP

Extraer en la raíz del proyecto `pfc1-stronghold-etapa1`:

- `Dockerfile.wikitext103`: imagen de datos basada en `pfc-stronghold:ds058`, con `pyarrow` y `nltk`.
- `SHv0-adaptado/pfc_tools/preparar_wikitext103.py`: descarga y transforma WikiText-103 y llama a `tools/preprocess_data.py` **existente**, sin modificarlo.
- `README_ETAPA2.md`: estas instrucciones.

## Ejecución paso a paso (PowerShell)

Abrir PowerShell en `C:\PFC1\STRONHOLD\pfc1-stronghold-etapa1`.

**1. Construir imagen de preparación:**

```powershell
docker build -t pfc-stronghold:data -f Dockerfile.wikitext103 .
```

**2. Descargar el dataset completo y los archivos de GPT-2:**

```powershell
New-Item -ItemType Directory -Force .\pfc_data, .\resultados\STRONGHOLD | Out-Null
$fecha = Get-Date -Format "yyyyMMdd_HHmmss"
docker run --rm -v "${PWD}/SHv0-adaptado:/work" -v "${PWD}/pfc_data:/data" -w /work pfc-stronghold:data python -B pfc_tools/preparar_wikitext103.py --stage download --root /data/wikitext103 2>&1 | Tee-Object -FilePath ".\resultados\STRONGHOLD\descarga_wikitext103_$fecha.log"
```

La descarga original de Salesforce en S3 ya no es accesible; el script usa la revisión fija `b08601e04326c79dfdd32d625aee71d232d685c3` de `Salesforce/wikitext` en Hugging Face, la variante `wikitext-103-raw-v1`. Verifica la suma SHA-256 de los cuatro archivos Parquet publicados por Salesforce. Los dos archivos GPT-2 BPE vienen de la distribución pública original de OpenAI y se registra su hash. Los datos oficiales de validación y test se conservan separados, sin incorporarlos al corpus de entrenamiento.

Archivo de auditoría: `pfc_data/wikitext103/01_descargas.json`.

**3. Convertir Parquet a JSONL (una entrada no vacía por fila original):**

```powershell
$fecha = Get-Date -Format "yyyyMMdd_HHmmss"
docker run --rm --network none -v "${PWD}/SHv0-adaptado:/work" -v "${PWD}/pfc_data:/data" -w /work pfc-stronghold:data python -B pfc_tools/preparar_wikitext103.py --stage convert --root /data/wikitext103 2>&1 | Tee-Object -FilePath ".\resultados\STRONGHOLD\conversion_wikitext103_$fecha.log"
```

Archivo de auditoría: `pfc_data/wikitext103/02_conversion.json`.

**4. Tokenizar y construir `.bin`/`.idx` con Megatron-LM original:**

```powershell
$fecha = Get-Date -Format "yyyyMMdd_HHmmss"
docker run --rm --network none -v "${PWD}/SHv0-adaptado:/work" -v "${PWD}/pfc_data:/data" -w /work pfc-stronghold:data python -B pfc_tools/preparar_wikitext103.py --stage index --root /data/wikitext103 --workers 4 2>&1 | Tee-Object -FilePath ".\resultados\STRONGHOLD\indice_wikitext103_$fecha.log"
```

La etapa `index` puede durar bastante porque tokeniza todo el corpus; espera a que finalice y comprueba `INDICES BIN/IDX MEGATRON: OK`.

Archivos resultantes:

```text
pfc_data/wikitext103/
├── raw_parquet/              # datos originales descargados, sin alterar
├── tokenizer/                # gpt2-vocab.json y gpt2-merges.txt
├── jsonl/                    # train, validation, test separados
├── megatron/
│   ├── wikitext103_gpt2_text_document.bin
│   └── wikitext103_gpt2_text_document.idx
├── 01_descargas.json
├── 02_conversion.json
└── 03_indice_megatron.json
```

El `--data-path` del entrenamiento será `/data/wikitext103/megatron/wikitext103_gpt2_text_document`. Usaremos los mismos datos e hiperparámetros para STRONGHOLD y Megatron sin offloading.

**No ejecutes los scripts `examples/pretrain_gpt_1gpu_*.sh` originales directamente:** incluyen borrado de archivos y modificaciones globales de librerías. Prepararemos lanzadores adaptados que solo configuran rutas y compatibilidad, sin alterar la técnica.

## Reproducibilidad y alcance

Fuentes:

- Dataset: https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1
- Tokenizador: https://github.com/openai/gpt-2
- Pipeline de indexación: `SHv0-adaptado/tools/preprocess_data.py` (código Megatron de STRONGHOLD)

La parte del benchmark del paper es **rendimiento del entrenamiento** (tamaño de modelo entrenable y muestras/s), no calidad lingüística. Tras completar la preparación, haremos una ejecución breve del GPT con `--enable-gl` y su baseline sin offloading; entonces ajustaremos tamaño de modelo, ventana y duración de ensayos en función de RAM/VRAM medidas, sin inventar resultados.
