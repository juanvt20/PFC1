# Herramientas PDF aisladas

Entorno operativo: `C:\PFC1\LSP-OFFLOAD\.venv-pdf-linux`.
Python 3.11 de la imagen local PyTorch de desarrollo; PyMuPDF **1.26.7**.
El venv no hereda paquetes globales (`include-system-site-packages=false`)
y no es el entorno de entrenamiento. Requiere montarse en `/work` porque
los venv contienen rutas absolutas. La imagen base solo ejecuta el proceso:
no se instaló PyMuPDF en sus capas ni se modificaron sus tags.

El intento inicial con Python MSYS2 3.14.7 creó `.venv-pdf/` y no consiguió
instalar PyMuPDF: no había rueda compatible y la compilación de dependencias
falló por falta de Ninja/make. Se conserva ese intento local excluido; el
entorno que se usa es `.venv-pdf-linux/`. No se instaló software global.

## Comandos utilizados

Desde `C:\PFC1`, primer intento:

```powershell
python -m venv LSP-OFFLOAD/.venv-pdf
& './LSP-OFFLOAD/.venv-pdf/bin/python.exe' -m pip install PyMuPDF
```

Instalación que sí terminó correctamente:

```powershell
$dockerExe = 'C:\Users\juan\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe'
& $dockerExe run --rm --pull=never --mount 'type=bind,source=C:\PFC1\LSP-OFFLOAD,target=/work' -w /work pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel bash -lc 'python -m venv /work/.venv-pdf-linux && /work/.venv-pdf-linux/bin/python -m pip install --only-binary=:all: PyMuPDF==1.26.7'
```

Extracción/renderizado ejecutados con el mismo montaje, `--network none`
y estos argumentos del contenedor, en este orden de lectura:

```text
.venv-pdf-linux/bin/python tools/pdf_extract.py LSP-OFFLOAD.pdf --pages 1,2,3,4,5,6,7,8,9,10,11,12,13
.venv-pdf-linux/bin/python tools/pdf_extract.py "PFC (1).pdf"
.venv-pdf-linux/bin/python tools/pdf_extract.py "PFC (1).pdf" --pages 5,6,7,8
.venv-pdf-linux/bin/python tools/pdf_extract.py STRONGHOLD.pdf --pages all
```

El script `tools/pdf.ps1` permite repetir estas operaciones sin montar
otros proyectos. Por ejemplo:

```powershell
./LSP-OFFLOAD/tools/pdf.ps1 -Action version
./LSP-OFFLOAD/tools/pdf.ps1 -Action extract -FileName LSP-OFFLOAD.pdf -Pages '4,5,6,7,11,12'
```

`tools/pdf_extract.py` usa `get_text(sort=True)` por página y
`get_pixmap(Matrix(1.5,1.5))` para PNG. Guarda manifiesto con versión,
SHA-256, número de páginas y cantidad de caracteres. No altera el PDF.
Los tres PDF tienen texto; no se ejecutó OCR. La extracción desordena
tablas giradas y mezcla columnas/etiquetas de figuras; los elementos
relevantes se contrastaron visualmente según CONTRASTE_PDF_CODIGO.md.

## Exclusiones

`referencias/`, `.venv-pdf/`, `.venv-pdf-linux/` y `tools/pdf_output/`
están excluidos de Git mediante `LSP-OFFLOAD/.gitignore`. No se copiaron
PDF ni páginas renderizadas al contexto Docker de entrenamiento:
`.dockerignore` permite únicamente requisitos y las cuatro fuentes del kernel.
