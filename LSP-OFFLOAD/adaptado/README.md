# Adaptación aislada: conservación de momentos

`patches/0001-preserve-transported-moments.patch` es el único cambio aplicado
al upstream en la copia local `upstream/`, excluida de Git. Corrige N15 con
la autorización del usuario: conserva M/V ya transportados al preparar
buffers y los devuelve al dispositivo del optimizador. No reinicia su
contador. La primera inicialización sigue creando ceros. Solo el compresor
LearnedCountSketch activa la nueva opción; la ruta SVD no se modifica.

Referencia: PDF LSP p.5, Algoritmo 1, líneas 8-9. El código almacena las
transpuestas de P/Q. Si A=P_nuevo^T P_anterior y B=Q_nuevo^T Q_anterior:
`M_nuevo=A M_anterior B^T`, `V_nuevo=A² V_anterior (B^T)²`, con cuadrados
elemento a elemento. La fórmula existente permanece intacta, igual que
epsilon, regularización, inicialización, muestreo y tratamiento de no finitos.

Reconstrucción desde `C:\PFC1`:

```powershell
python LSP-OFFLOAD/tools/prepare_adaptation.py
```

El preparador verifica los 99 blobs del manifiesto antes de copiar;
rechaza sobrescribir una copia adaptada que difiera de su salida esperada.
La copia original no se escribe. El parche también admite comprobación
con `git apply --check` desde el upstream usando su ruta absoluta.

Pruebas reproducibles en la imagen aislada existente:

```powershell
& ./LSP-OFFLOAD/tools/run_regression.ps1 -Suite moments -Source original -RunId moment-original-captured-inputs
& ./LSP-OFFLOAD/tools/run_regression.ps1 -Suite moments -Source adaptado -RunId moment-adapted-captured-inputs
& ./LSP-OFFLOAD/tools/run_regression.ps1 -Suite scheduler -Source adaptado -RunId scheduler-boundaries-before-fix
```

Esos IDs ya existen: usar otros para repetir sin sobrescribir evidencia.
La primera prueba falla como se espera en el original (exit1); la segunda
pasa (exit0). Verifica los tensores completos, dos recalibraciones,
inicialización, buffers y siguiente paso de Adam: CPU/GPU y pesos
float32/bfloat16, estados float32. Captura los sketches densos realmente
usados; no los regenera para evitar que colisiones cambien la referencia.
Referencia de transporte en float64 sobre los solapamientos con el redondeo
original; tolerancias rtol2e-5/atol2e-6. Los tensores completos se guardan
localmente en `.pt`; el resumen está en `configs/adaptation_validation.json`.

La prueba del planificador **falla** (exit1): confirma N14 en evaluación
y salida, tanto con Zero activado explícitamente como con una configuración
FCFS/LCFS. El control serial pasa. No se aplicó una corrección del scheduler;
el piloto y la comparación quedan detenidos por este bloqueo. Véase el
registro existente `docs/VALIDACION_LOCAL.md` para evidencia y propuesta.
