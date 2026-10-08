# pixi image on the CUDA 13.0 base (nvidia/cuda:13.0.0-base-ubuntu24.04).
# The base gives the CUDA runtime env + nvidia-container integration; nvcc for
# pykeops comes from the conda `cuda-nvcc` package (see pixi.toml).
FROM ghcr.io/prefix-dev/pixi:noble-cuda-13.0.0

WORKDIR /app/

ARG TRUSTED_CA_CERT_PATH=""
RUN \
    if [ -n "$TRUSTED_CA_CERT_PATH" -a -e "$TRUSTED_CA_CERT_PATH" ]; then \
    echo "Adding custom certificate to certificate store."; \
    cp "$TRUSTED_CA_CERT_PATH" /usr/local/share/ca-certificates/quaisr-additional-trusted-ca.crt; \
    update-ca-certificates; \
    fi

# Non-root user + writable pykeops JIT cache (KEOPS_CACHE_FOLDER=/tmp/keops).
# Created BEFORE the env install so files are owned by quaisr_user from the
# start. A `chown -R /app` after the install rewrites every file of the ~6 GB
# pixi env into a new layer, doubling the image size (and node pull time).
RUN groupadd -r quaisr_user && useradd -r -m -g quaisr_user quaisr_user && \
    mkdir -p /tmp/keops && \
    chown quaisr_user:quaisr_user /app /tmp/keops

COPY --chown=quaisr_user:quaisr_user . /app/

USER quaisr_user

# Refresh pixi.lock to match pixi.toml, then solve + install the full env
# (conda faiss-gpu + all PyPI deps).
# Clean the pixi download caches (conda tarballs + uv wheels, ~6-7 GB) in the
# same layer so they never land in a snapshot -- keeps kaniko peak storage and
# the final image smaller.
RUN pixi lock && pixi install && pixi clean cache --yes

# keops 2.3 detects CUDA via ctypes.util.find_library("cuda"/"nvrtc"), which
# resolves through the ldconfig cache and IGNORES LD_LIBRARY_PATH. The conda env
# lib dir is not on ldconfig's path, so find_library("nvrtc") returned None and
# keops fell back to CPU. Register the env lib dir (holds libnvrtc.so.13) plus the
# nvidia driver dirs with ldconfig. The nvidia container runtime regenerates the
# cache at container start and re-reads this conf file, so the host-mounted
# libcuda.so.1 (present only at run) gets picked up for find_library("cuda") too.
# Writes to /etc, so it runs as root.
USER root
RUN printf '%s\n' \
      /app/.pixi/envs/default/lib \
      /usr/local/nvidia/lib64 \
      /usr/local/nvidia/lib \
    > /etc/ld.so.conf.d/quaisr-cuda.conf && ldconfig
USER quaisr_user

# quaisr runtime pulled from the private index. Installed WITH deps so pydantic
# and the rest of quaisr's runtime deps land in the env; pixi already pinned the
# heavy GPU packages (torch/faiss) so pip only fills in the light pure-python gaps.
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
RUN pixi run pip install quaisr \
        --index-url $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST

# Run as a module (-m) from /app so the namespace package (no __init__.py) is on
# sys.path; invoking the file by path would only add the script's own dir.
ENTRYPOINT ["pixi", "run", "python", "-m", "nearest_neighbour_gaussian_process"]
