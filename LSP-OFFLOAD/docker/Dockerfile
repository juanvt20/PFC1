FROM pytorch/pytorch@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9
LABEL org.opencontainers.image.title="PFC1 LSP-Offload isolated validation"
ENV MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=12.0 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
COPY docker/requirements.txt docker/constraints.txt /opt/lsp/
RUN python -m pip install --no-cache-dir -r /opt/lsp/requirements.txt -c /opt/lsp/constraints.txt && \
    python -m pip check && \
    python -c "import torch; assert torch.__version__.startswith('2.7.1'); assert torch.version.cuda == '12.8'" && \
    python -m pip freeze > /opt/lsp/environment.freeze.txt
# Copy only four original source files. Never reuse the published CPython 3.10 binary.
COPY original/upstream/kernels/setup.py original/upstream/kernels/fusedAdam.cpp original/upstream/kernels/fusedAdam_cpu.cpp original/upstream/kernels/fusedAdam_cuda.cu /opt/lsp/kernels/
WORKDIR /opt/lsp/kernels
RUN python setup.py build_ext --inplace
ENV PYTHONPATH=/opt/lsp
WORKDIR /work
