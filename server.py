#!/usr/bin/env python3
"""OpenAI-compatible TTS API for one multi-speaker Qwen3 fine-tune.

  GET  /health
  GET  /v1/models
  GET  /v1/voices
  POST /v1/audio/speech   {input, voice, model?, instructions?} -> WAV
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

HOST = os.environ.get("TTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("TTS_PORT", "8080"))
MODEL_PATH = os.environ.get("TTS_MODEL", os.environ.get("MODEL_PATH", "/models"))
VOICES_PATH = Path(os.environ.get("TTS_VOICES", "/config/voices.json"))
DEFAULT_LANGUAGE = os.environ.get("TTS_LANGUAGE", "English")
MODEL_NAME = os.environ.get("TTS_MODEL_NAME", "tts-1")


def pick_device() -> str:
    requested = os.environ.get("TTS_DEVICE", "").strip()
    if requested:
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


DEVICE = pick_device()

app = FastAPI(title="Qwen3 TTS OpenAI")
lock = threading.Lock()
engine = None
speakers: dict[str, str] = {}
default_voice = ""
ready_error: str | None = None


class OpenAISpeechRequest(BaseModel):
    input: str = ""
    text: str = ""
    voice: str = ""
    model: str = ""
    instructions: str | None = None
    language: str | None = None
    response_format: str = "wav"
    speed: float | None = None


def speaker_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def load_voice_map(supported: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if VOICES_PATH.is_file():
        data = json.loads(VOICES_PATH.read_text(encoding="utf-8"))
        voices = data.get("voices", data)
        if isinstance(voices, dict):
            for name, spec in voices.items():
                if isinstance(spec, str):
                    mapping[name.lower()] = spec
                elif isinstance(spec, dict):
                    mapping[str(name).lower()] = str(spec.get("speaker") or spec.get("name") or name)
        elif isinstance(voices, list):
            for name in voices:
                mapping[str(name).lower()] = str(name)
    env_speakers = os.environ.get("TTS_SPEAKERS", "").strip()
    if env_speakers:
        for name in env_speakers.split(","):
            name = name.strip()
            if name:
                mapping[name.lower()] = name
    for spk in supported:
        mapping.setdefault(str(spk).lower(), str(spk))
    if not mapping:
        mapping["default"] = "default"
    return mapping


def resolve_voice(name: str | None) -> str:
    key = (name or "").strip().lower()
    if not key:
        return default_voice
    if key in speakers:
        return speakers[key]
    wanted = speaker_key(key)
    for alias, real in speakers.items():
        if speaker_key(alias) == wanted or speaker_key(real) == wanted:
            return real
    raise HTTPException(status_code=400, detail=f"unknown voice {name!r}; known: {sorted(speakers)}")


def ensure_tokenizer_weights(model_dir: Path) -> None:
    tok = model_dir / "speech_tokenizer"
    if not tok.is_dir():
        return
    if (tok / "model.safetensors").exists() or (tok / "pytorch_model.bin").exists():
        return
    extra = os.environ.get("TTS_TOKENIZER", "").strip()
    candidates = [Path(p) for p in extra.split(os.pathsep) if p.strip()]
    candidates.extend(
        [
            model_dir.parent / "speech_tokenizer" / "model.safetensors",
            Path("/tokenizer/model.safetensors"),
        ]
    )
    for src in candidates:
        if src.is_file():
            dest = tok / "model.safetensors"
            dest.write_bytes(src.read_bytes())
            return


@app.on_event("startup")
def startup() -> None:
    global engine, speakers, default_voice, ready_error
    from qwen_tts import Qwen3TTSModel

    model_dir = Path(MODEL_PATH)
    if not model_dir.exists():
        ready_error = f"model path missing: {model_dir}"
        raise RuntimeError(ready_error)
    ensure_tokenizer_weights(model_dir)
    use_cuda = str(DEVICE).startswith("cuda")
    engine = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map=DEVICE,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        attn_implementation="sdpa" if use_cuda else "eager",
    )
    supported = []
    if hasattr(engine, "get_supported_speakers"):
        supported = list(engine.get_supported_speakers() or [])
    speakers = load_voice_map(supported)
    env_default = os.environ.get("TTS_DEFAULT_VOICE", "").strip().lower()
    default_voice = speakers.get(env_default) if env_default in speakers else next(iter(speakers.values()))
    print(f"device={DEVICE} voices={sorted(speakers)} default={default_voice}", flush=True)


@app.get("/health")
def health():
    if engine is None:
        return {"ok": False, "error": ready_error or "loading"}
    return {
        "ok": True,
        "voices": sorted(set(speakers.values())),
        "default": default_voice,
        "device": DEVICE,
        "model": str(MODEL_PATH),
    }


@app.get("/v1/voices")
def openai_voices():
    names = sorted(set(speakers.values()))
    return {
        "object": "list",
        "data": [{"voice_id": n, "name": n} for n in names],
        "default": default_voice,
    }


@app.get("/v1/models")
def openai_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAME, "object": "model", "owned_by": "local"},
            {"id": "qwen3-tts", "object": "model", "owned_by": "local"},
        ],
    }


@app.post("/v1/audio/speech")
def openai_speech(req: OpenAISpeechRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail=ready_error or "loading")
    text = (req.input or req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty input")
    fmt = (req.response_format or "wav").lower()
    if fmt not in {"wav", "pcm"}:
        raise HTTPException(status_code=400, detail="only wav and pcm are supported")
    speaker = resolve_voice(req.voice)
    language = req.language or DEFAULT_LANGUAGE
    with lock:
        wavs, sr = engine.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=req.instructions,
        )
        audio = np.asarray(wavs[0], dtype=np.float32)
    if fmt == "pcm":
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        return Response(content=pcm, media_type="application/octet-stream", headers={"X-TTS-Voice-Used": speaker})
    return Response(content=wav_bytes(audio, sr), media_type="audio/wav", headers={"X-TTS-Voice-Used": speaker})


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
