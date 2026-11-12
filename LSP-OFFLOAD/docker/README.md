# Diseño del entorno: Etapa B pendiente

Se conservará el entorno de STRONGHOLD y se construirá una imagen independiente para LSP-Offload. No se instaló software en la laptop desde esta sesión.

Base candidata: PyTorch2.7.1 + CUDA12.8 + cuDNN, variante **de desarrollo**, con nvcc. Verificar existencia del tag y fijar digest al construir; el tag runtime usado antes no basta para compilar el CUDAExtension original. El entorno final incluirá compilador C++, OpenMP/libgomp, setuptools y ninja, y versiones resueltas de las dependencias de la ruta prioritaria.

Compilar con `TORCH_CUDA_ARCH_LIST=12.0` y `MAX_JOBS=1` dentro de la imagen. Confirmar nvcc, arquitectura y versión de Python efectiva; no reutilizar el `.so` del upstream. Los kernels CPU requieren float32, AVX2 y tamaños múltiplo de8. Validar la fórmula Adam de cada dispositivo según el registro de compatibilidad.

No instalar drivers Linux de NVIDIA en WSL ni modificar paquetes globales. No ejecutar todavía los scripts upstream completos. No cambiar `.wslconfig`, asignación de RAM, swap ni otros proyectos en esta etapa.

Datos/modelos/resultados se montarán en rutas separadas; el código original se podrá montar de solo lectura. La imagen y todo comando deben guardar versiones y salida de compilación. Evitar descargas duplicadas y compilar sin modelos cargados en memoria. Los binarios de compilación serán derivados locales, no contenido Git.
