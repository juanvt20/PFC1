# Código original y procedencia

Commit fijado: `d9dc92a32b7e4829116ba6abec17c280dfb548a4`.

Repositorio: https://github.com/gulang2019/LSP-Offload.

`UPSTREAM.json` registra todos los archivos del commit con sus hashes Git. Esta entrega no incorpora modelos, datasets, resultados de referencia ni binarios de los autores.

`original/upstream/` ya contiene la copia local completa de ese commit y está excluido de Git del PFC. `python LSP-OFFLOAD/tools/verify_upstream.py` comprueba los 99 archivos contra el manifiesto y sus objetos Git. No se usaron los binarios publicados. El estado de Git en Windows puede mostrar cambios por expectativas CRLF/estadísticas del índice; el hash de bytes y `git -c core.autocrlf=false diff --quiet` distinguen contenido de representación. No normalizar archivos para obtener un estado visualmente limpio.

El árbol auditado no contiene un archivo LICENSE/COPYING ni se ha identificado una concesión explícita general. Conservar todos los avisos/atribuciones existentes. Versionar las adaptaciones propias como parches y configuraciones separados; revisar el permiso de redistribución antes de copiar el código de terceros al repositorio público. No se atribuye licencia MIT/Apache por inferencia.
