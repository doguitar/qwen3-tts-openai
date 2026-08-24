FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
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
        python3-venv \
        ffmpeg \
        libsndfile1 \
        sox \
        libsox-dev \
        ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install CUDA torch first so pip does not replace it with a CPU wheel.
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir \
        torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir -r requirements.txt

COPY server.py /app/server.py
COPY voices.example.json /app/voices.example.json

EXPOSE 8080
VOLUME ["/models", "/config"]
CMD ["python3", "/app/server.py"]
