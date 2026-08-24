FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=8080 \
    TTS_MODEL=/models \
    TTS_VOICES=/config/voices.json \
    TTS_LANGUAGE=English \
    TTS_MODEL_NAME=tts-1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        sox \
        ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# CUDA kernels live in the torch wheels. Drop the nvidia/cuda base (duplicate
# ~700MB) and skip Gradio (qwen-tts UI, not used by this API).
RUN pip3 install --upgrade pip \
    && pip3 install \
        torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 uninstall -y nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-cuda-cupti-cu12 || true

RUN pip3 install --no-deps qwen-tts==0.1.1 \
    && pip3 install -r requirements.txt \
    && pip3 uninstall -y gradio gradio-client hf-gradio groovy || true \
    && find /usr/local/lib -type d -name __pycache__ -exec rm -rf {} + \
    && find /usr/local/lib -type d -name tests -exec rm -rf {} + \
    && find /usr/local/lib -type d -name test -exec rm -rf {} + \
    && rm -rf /root/.cache /tmp/*

COPY server.py /app/server.py
COPY voices.example.json /app/voices.example.json

EXPOSE 8080
VOLUME ["/models", "/config"]
CMD ["python3", "/app/server.py"]
