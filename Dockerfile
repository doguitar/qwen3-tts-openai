FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=8080 \
    TTS_MODEL=/models \
    TTS_VOICES=/config/voices.json \
    TTS_LANGUAGE=English \
    TTS_MODEL_NAME=tts-1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py /app/server.py
COPY voices.example.json /app/voices.example.json

EXPOSE 8080
VOLUME ["/models", "/config"]
CMD ["python", "/app/server.py"]
