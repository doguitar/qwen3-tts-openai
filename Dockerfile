ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG TORCH_BACKEND=cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=8080 \
    TTS_MODEL=/models \
    TTS_VOICES=/config/voices.json \
    TTS_LANGUAGE=English \
    TTS_MODEL_NAME=tts-1 \
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

# Intel Level Zero / OpenCL userspace for Arc (A380 Alchemist and later).
# Host still needs the i915/xe kernel driver; pass --device /dev/dri at runtime.
RUN if [ "$TORCH_BACKEND" = "xpu" ]; then \
        . /etc/os-release; \
        wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
            | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
        && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME} client" \
            > /etc/apt/sources.list.d/intel-gpu.list \
        || true; \
        apt-get update; \
        apt-get install -y --no-install-recommends clinfo intel-opencl-icd || true; \
        apt-get install -y --no-install-recommends libze1 || true; \
        apt-get install -y --no-install-recommends libze-intel-gpu1 intel-level-zero-gpu level-zero || true; \
        rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app
COPY requirements.txt .

# CUDA kernels live in the torch wheels. Drop the nvidia/cuda base (duplicate
# ~700MB) and skip Gradio (qwen-tts UI, not used by this API).
# XPU: official PyTorch Intel GPU wheels (Arc A-series, including A380).
RUN pip3 install --upgrade pip \
    && if [ "$TORCH_BACKEND" = "xpu" ]; then \
        pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/xpu; \
    elif [ "$TORCH_BACKEND" = "cpu" ]; then \
        pip3 install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip3 install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
        && pip3 uninstall -y nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-cuda-cupti-cu12 || true; \
    fi

RUN pip3 install --no-deps qwen-tts==0.1.1 \
    && pip3 install -r requirements.txt \
    && pip3 uninstall -y gradio gradio-client hf-gradio groovy || true \
    && find /usr/local/lib -type d \( -name __pycache__ -o -name tests -o -name test \) -prune -exec rm -rf {} + || true \
    && rm -rf /root/.cache /tmp/*

COPY server.py /app/server.py
COPY device.py /app/device.py
COPY voices.example.json /app/voices.example.json

EXPOSE 8080
VOLUME ["/models", "/config"]
CMD ["python3", "/app/server.py"]
