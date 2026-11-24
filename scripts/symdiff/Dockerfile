# PatchSymDiff image: compares a function in an original and a patched binary
# by symbolic execution (scripts/patch-runtime/patch_symdiff.py, on cozy).
#
# The script needs its own environment, because cozy pins an older angr than
# the rest of the validation scripts use.  This image is that environment.
#
# Build context: the repository's scripts/ directory.
#
#   docker build -t patchestry/symdiff:latest -f scripts/symdiff/Dockerfile scripts
#   docker run --rm -v /path/to/binaries:/mnt/input:ro patchestry/symdiff:latest \
#       --original /mnt/input/original.elf --rewritten /mnt/input/patched.elf \
#       --function NAME
#
# Use scripts/symdiff/build-symdiff-docker.sh and run-symdiff.sh, which do
# the mounting.
#
# Stages: `runtime` is the image.  `test` adds clang and lld and runs the
# script's unit tests in the image's environment (build it with --target test).

ARG PYTHON_VERSION=3.11

# ============================================================
# Stage 1: install the pinned environment
# ============================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# A compiler and CMake for the packages that have no wheel for the platform
# (unicorn on some).  They stay in this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

COPY symdiff/requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================
# Stage 2: the image
# ============================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

COPY --from=builder /opt/venv /opt/venv
COPY patch-runtime/patch_symdiff.py /opt/symdiff/patch_symdiff.py

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore

# Fail the build if the environment cannot load what the script imports.
RUN python -c "import angr, cozy, claripy, elftools, unicorn"

WORKDIR /work
ENTRYPOINT ["python", "/opt/symdiff/patch_symdiff.py"]
CMD ["--help"]

# ============================================================
# Stage 3: the unit tests, in the image's environment
# ============================================================
FROM runtime AS test

RUN apt-get update && apt-get install -y --no-install-recommends clang lld \
    && rm -rf /var/lib/apt/lists/*

# The tests look for the script beside themselves.
COPY patch-runtime/patch_symdiff.py patch-runtime/test_patch_symdiff.py /opt/symdiff/
ENV PATCHESTRY_LLVM_PREFIX=/usr/bin
RUN cd /opt/symdiff && python -W ignore -m unittest -v test_patch_symdiff 2>&1 | tee /opt/symdiff/test.log

ENTRYPOINT ["cat", "/opt/symdiff/test.log"]
CMD []
