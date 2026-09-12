ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

# Stable for all backends. Do not declare TORCH_BACKEND above this block —
# BuildKit includes declared ARGs in the cache key of every later layer.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    ZE_FLAT_DEVICE_HIERARCHY=FLAT \
    SYCL_CACHE_PERSISTENT=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        sox \
        ca-certificates \
        wget \
        gnupg \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Shared Python deps (no torch). cpu / cuda / xpu reuse this layer.
RUN pip3 install --upgrade pip \
    && pip3 install --no-deps qwen-tts==0.1.1 \
    && pip3 install -r requirements.txt \
    && pip3 uninstall -y gradio gradio-client hf-gradio groovy || true \
    && find /usr/local/lib -type d \( -name __pycache__ -o -name tests -o -name test \) -prune -exec rm -rf {} + || true \
    && rm -rf /root/.cache /tmp/*

# Backend-specific from here down.
ARG TORCH_BACKEND=cpu

# Intel Level Zero / OpenCL userspace for Arc (A380 Alchemist and later).
# Host still needs the i915/xe kernel driver and Resizable BAR; pass --device=/dev/dri.
# libze-intel-gpu1 Breaks: intel-level-zero-gpu — install only the former.
# Use the Ubuntu "unified" channel, not "client": jammy client ships compute-runtime
# 24.39, which cannot create a oneDNN GPU engine with torch 2.13+xpu
# (RuntimeError: could not make an engine with allocator). unified 25.18 works.
RUN if [ "$TORCH_BACKEND" = "xpu" ]; then \
        . /etc/os-release; \
        wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
            | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
        && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME} unified" \
            > /etc/apt/sources.list.d/intel-gpu.list \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
            clinfo \
            intel-opencl-icd \
            libze1 \
            libze-intel-gpu1 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Torch wheels: cpu (default, :latest), cuda (cu124), or xpu (Intel Arc).
# CUDA kernels live in the torch wheels — no nvidia/cuda base image.
RUN if [ "$TORCH_BACKEND" = "xpu" ]; then \
        pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/xpu; \
    elif [ "$TORCH_BACKEND" = "cpu" ]; then \
        pip3 install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip3 install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
        && pip3 uninstall -y nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-cuda-cupti-cu12 || true; \
    fi \
    && rm -rf /root/.cache /tmp/*

ENV TTS_HOST=0.0.0.0 \
    TTS_PORT=8080 \
    TTS_MODEL=/models \
    TTS_VOICES=/config/voices.json \
    TTS_LANGUAGE=English \
    TTS_MODEL_NAME=tts-1

COPY server.py device.py models.py voices.example.json /app/
COPY static /app/static

EXPOSE 8080
VOLUME ["/models", "/config"]
CMD ["python3", "/app/server.py"]
