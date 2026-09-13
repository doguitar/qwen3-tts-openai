#!/usr/bin/env python3
"""OpenAI-compatible TTS API for one or more Qwen3 fine-tunes.

  GET  /
  GET  /config
  GET  /health
  GET  /v1/models          one public id (TTS_MODEL_NAME, default tts-1)
  GET  /v1/voices          `{folder}-{speaker}` for every checkpoint
  POST /v1/audio/speech    voice selects the checkpoint
"""
from __future__ import annotations

import asyncio
import gc
import io
import json
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

# Level Zero must be configured before torch.xpu initializes (Arc A380 / Alchemist).
os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, field_validator

from device import inference_settings, select_device
from models import (
    VoiceOverlay,
    build_voice_index,
    checkpoint_kind,
    checkpoint_speakers,
    default_model_id,
    discover_checkpoints,
    is_public_model_request,
    load_voices_document,
    merge_instructions,
    overlay_clone_ref,
    overlay_instructions,
    overlay_kind,
    parse_load_policy,
    parse_voice_overlays,
    public_default_voice,
    public_voice_id,
    public_voice_names,
    resolve_overlay_target,
    resolve_voice_route,
    validate_voices_document,
    voices_file_writable,
    write_voices_document,
)

OPENAI_STOCK_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
}

AUDIO_FORMATS = {
    "wav": ("audio/wav", ["-f", "wav", "-acodec", "pcm_s16le"]),
    "mp3": ("audio/mpeg", ["-f", "mp3", "-acodec", "libmp3lame", "-q:a", "2"]),
    "opus": ("audio/opus", ["-f", "opus"]),
    "aac": ("audio/aac", ["-f", "adts", "-acodec", "aac"]),
    "flac": ("audio/flac", ["-f", "flac"]),
    "pcm": ("application/octet-stream", None),
}

HOST = os.environ.get("TTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("TTS_PORT", "8080"))
MODEL_PATH = os.environ.get("TTS_MODEL", os.environ.get("MODEL_PATH", "/models"))
VOICES_PATH = Path(os.environ.get("TTS_VOICES", "/config/voices.json"))
CLONES_DIR = VOICES_PATH.parent / "clones"
DEFAULT_LANGUAGE = os.environ.get("TTS_LANGUAGE", "English")
MODEL_NAME = os.environ.get("TTS_MODEL_NAME", "tts-1")
LOAD_POLICY = parse_load_policy(os.environ.get("TTS_LOAD_POLICY", ""))
TTS_DEFAULT_MODEL = os.environ.get("TTS_DEFAULT_MODEL", "").strip()
STATIC_DIR = Path(__file__).resolve().parent / "static"


def xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    try:
        return bool(xpu is not None and xpu.is_available())
    except Exception:
        return False


def pick_device() -> str:
    return select_device(os.environ.get("TTS_DEVICE", ""), torch.cuda.is_available(), xpu_available())


def xpu_device_name(device: str) -> str | None:
    if not str(device).startswith("xpu") or not xpu_available():
        return None
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    return torch.xpu.get_device_name(index)


DEVICE = pick_device()
DTYPE_NAME, ATTN = inference_settings(DEVICE, os.environ.get("TTS_DTYPE", ""))
DTYPE = getattr(torch, DTYPE_NAME)

app = FastAPI(title="Qwen3 TTS OpenAI")
lock = threading.Lock()
catalog: list[tuple[str, Path]] = []
default_id = ""
loaded: dict[str, Any] = {}
speakers_by: dict[str, dict[str, str]] = {}
default_voice_by: dict[str, str] = {}
kind_by: dict[str, str] = {}
voice_index: dict[str, tuple[str, str]] = {}
voice_overlays: list[VoiceOverlay] = []
default_voice = ""
ready_error: str | None = None
BODY_LOG_LIMIT = int(os.environ.get("TTS_LOG_BODY_LIMIT", "8000"))


def preview_body(raw: bytes | None) -> str:
    text = (raw or b"").decode("utf-8", "replace")
    if len(text) > BODY_LOG_LIMIT:
        return text[:BODY_LOG_LIMIT] + "...(truncated)"
    return text


def log_api_error(request: Request, status: int, detail: Any) -> None:
    raw = getattr(request.state, "raw_body", b"")
    print(
        f"API error {request.method} {request.url.path} status={status} "
        f"detail={detail!r} body={preview_body(raw)!r}",
        flush=True,
    )


@app.middleware("http")
async def capture_request_body(request: Request, call_next):
    raw = await request.body()

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    request = Request(request.scope, receive)
    request.state.raw_body = raw
    try:
        response = await call_next(request)
    except Exception as exc:
        log_api_error(request, 500, repr(exc))
        raise
    if response.status_code >= 400:
        log_api_error(request, response.status_code, f"http {response.status_code}")
    return response


class OpenAISpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: str = ""
    text: str = ""
    voice: Any = ""
    model: str = ""
    instructions: str | None = None
    language: str | None = None
    response_format: str = "mp3"
    speed: float | None = None
    stream_format: str | None = None

    @field_validator("voice", mode="before")
    @classmethod
    def coerce_voice(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            return str(value.get("id") or value.get("voice") or value.get("name") or "")
        return str(value)


def speaker_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _read_overlays() -> list[VoiceOverlay]:
    data = None
    if VOICES_PATH.is_file():
        data = json.loads(VOICES_PATH.read_text(encoding="utf-8"))
    return parse_voice_overlays(data, os.environ.get("TTS_SPEAKERS", ""))


def resolve_voice(name: str | None, speakers: dict[str, str], model_default: str) -> tuple[str, bool, str]:
    key = (name or "").strip().lower()
    if not key or key in OPENAI_STOCK_VOICES:
        reason = "empty voice" if not key else f"openai stock voice {key}"
        return model_default, bool(key), reason
    if key in speakers:
        return speakers[key], False, ""
    prefix = ""
    for mid in catalog_ids():
        token = f"{mid}-"
        if key.startswith(token.lower()):
            prefix = token
            break
    if prefix:
        stripped = key[len(prefix) :]
        if stripped in speakers:
            return speakers[stripped], False, ""
    wanted = speaker_key(key)
    for alias, real in speakers.items():
        if speaker_key(alias) == wanted or speaker_key(real) == wanted:
            return real, False, ""
    return model_default, True, f"unknown voice {name!r}; using {model_default}"


def encode_audio(audio: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    fmt = (fmt or "mp3").lower().strip()
    if fmt not in AUDIO_FORMATS:
        fmt = "mp3"
    media, ffmpeg_args = AUDIO_FORMATS[fmt]
    wav = wav_bytes(audio, sr)
    if fmt == "wav":
        return wav, media
    if fmt == "pcm":
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        return pcm, media
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *ffmpeg_args, "pipe:1"],
        input=wav,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        print(f"ffmpeg {fmt} failed: {proc.stderr.decode('utf-8', 'replace')}", flush=True)
        return wav, "audio/wav"
    return proc.stdout, media


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
            try:
                dest.write_bytes(src.read_bytes())
            except OSError:
                # /models is often mounted :ro; HF may still resolve sibling/parent.
                return
            return


def catalog_ids() -> list[str]:
    return [i for i, _ in catalog]


def _catalog_path(model_id: str) -> Path:
    for i, path in catalog:
        if i == model_id:
            return path
    raise KeyError(model_id)


def _rebuild_voice_index() -> None:
    global voice_index, default_voice
    voice_index = build_voice_index(catalog, voice_overlays, default_id)
    default_voice = public_default_voice(
        voice_index,
        os.environ.get("TTS_DEFAULT_VOICE", ""),
        catalog_ids(),
        voice_overlays,
    )



def _reload_voice_overlays() -> None:
    global voice_overlays
    voice_overlays = _read_overlays()
    _rebuild_voice_index()


def _refresh_speakers_maps() -> None:
    ids = set(catalog_ids())
    for stale in list(speakers_by):
        if stale not in ids:
            speakers_by.pop(stale, None)
            default_voice_by.pop(stale, None)
    for stale in list(kind_by):
        if stale not in ids:
            kind_by.pop(stale, None)
    for model_id, path in catalog:
        supported = checkpoint_speakers(path)
        if model_id in loaded and hasattr(loaded[model_id], "get_supported_speakers"):
            extra = list(loaded[model_id].get_supported_speakers() or [])
            if extra:
                supported = extra
        speakers_by[model_id], default_voice_by[model_id] = _speakers_for(model_id, supported)
        kind_by[model_id] = checkpoint_kind(path)


def _rescan_catalog() -> dict:
    global catalog, default_id
    with lock:
        found = discover_checkpoints(Path(MODEL_PATH), MODEL_NAME)
        if not found:
            raise HTTPException(status_code=400, detail=f"no checkpoints under {MODEL_PATH}")
        new_ids = {i for i, _ in found}
        for mid in list(loaded):
            if mid not in new_ids:
                _unload_one(mid)
        catalog = found
        default_id = default_model_id(catalog_ids(), TTS_DEFAULT_MODEL)
        _refresh_speakers_maps()
        _reload_voice_overlays()
        print(
            f"rescanned public={MODEL_NAME} voices={public_voice_names(voice_index, voice_overlays, catalog)} "
            f"checkpoints={catalog_ids()} default_ckpt={default_id}",
            flush=True,
        )
        return {
            "ok": True,
            "voices": public_voice_names(voice_index, voice_overlays, catalog),
            "default": default_voice,
            "models": [
                {"id": i, "path": str(p), "loaded": i in loaded, "kind": kind_by.get(i, "custom_voice")}
                for i, p in catalog
            ],
        }


def _voices_ui_payload() -> dict:
    document, error = load_voices_document(VOICES_PATH)
    return {
        "path": str(VOICES_PATH),
        "exists": VOICES_PATH.is_file(),
        "writable": voices_file_writable(VOICES_PATH),
        "document": document,
        "speakers_env": os.environ.get("TTS_SPEAKERS", ""),
        "error": error,
    }


def _speakers_for(model_id: str, supported: list[str]) -> tuple[dict[str, str], str]:
    mapping: dict[str, str] = {}
    for name in supported:
        mapping[str(name).lower()] = str(name)
    for item in voice_overlays:
        if item.kind == "voice_clone":
            continue
        resolved = resolve_overlay_target(item.speaker, item.model, voice_index, catalog_ids(), default_id)
        if resolved and resolved[0] == model_id:
            mapping[item.alias.lower()] = resolved[1]
    if not mapping:
        mapping[model_id.lower()] = model_id
    env_default = os.environ.get("TTS_DEFAULT_VOICE", "").strip().lower()
    model_default = mapping[env_default] if env_default in mapping else next(iter(mapping.values()))
    return mapping, model_default


def _load_one(model_id: str) -> None:
    if model_id in loaded:
        return
    path = _catalog_path(model_id)
    ensure_tokenizer_weights(path)
    from qwen_tts import Qwen3TTSModel

    engine = Qwen3TTSModel.from_pretrained(
        str(path),
        device_map=DEVICE,
        torch_dtype=DTYPE,
        attn_implementation=ATTN,
    )
    supported = checkpoint_speakers(path)
    if hasattr(engine, "get_supported_speakers"):
        extra = list(engine.get_supported_speakers() or [])
        if extra:
            supported = extra
    speakers_by[model_id], default_voice_by[model_id] = _speakers_for(model_id, supported)
    kind_by[model_id] = checkpoint_kind(path)
    loaded[model_id] = engine
    _rebuild_voice_index()
    print(
        f"loaded model={model_id!r} path={path} device={DEVICE} dtype={DTYPE_NAME} attn={ATTN} "
        f"voices={sorted(speakers_by[model_id])} default={default_voice_by[model_id]}",
        flush=True,
    )


def _unload_one(model_id: str) -> None:
    engine = loaded.pop(model_id, None)
    if engine is None:
        return
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if xpu_available() and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()
    print(f"unloaded model={model_id!r}", flush=True)


def _ensure_engine(model_id: str) -> Any:
    if LOAD_POLICY == "one" and set(loaded) != {model_id}:
        for mid in list(loaded):
            _unload_one(mid)
    _load_one(model_id)
    return loaded[model_id]


@app.on_event("startup")
def startup() -> None:
    global catalog, default_id, ready_error, voice_overlays
    catalog = discover_checkpoints(Path(MODEL_PATH), MODEL_NAME)
    if not catalog:
        ready_error = f"no checkpoints under {MODEL_PATH}"
        raise RuntimeError(ready_error)
    default_id = default_model_id(catalog_ids(), TTS_DEFAULT_MODEL)
    voice_overlays = _read_overlays()
    _rebuild_voice_index()
    try:
        if LOAD_POLICY == "all":
            for model_id, _ in catalog:
                _load_one(model_id)
        elif LOAD_POLICY == "one":
            _load_one(default_id)
    except Exception as exc:
        ready_error = str(exc)
        raise
    _refresh_speakers_maps()
    xpu_name = xpu_device_name(DEVICE)
    print(
        f"policy={LOAD_POLICY} public={MODEL_NAME} voices={public_voice_names(voice_index, voice_overlays, catalog)} "
        f"default_voice={default_voice} checkpoints={catalog_ids()} default_ckpt={default_id} "
        f"device={DEVICE} dtype={DTYPE_NAME} attn={ATTN} xpu={xpu_name!r}",
        flush=True,
    )



@app.get("/")
def ui_index():
    path = STATIC_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ui missing")
    return FileResponse(path)


@app.get("/design")
def ui_design():
    path = STATIC_DIR / "design.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ui missing")
    return FileResponse(path)


@app.get("/clone")
def ui_clone():
    path = STATIC_DIR / "clone.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ui missing")
    return FileResponse(path)


@app.get("/config")
def ui_config():
    path = STATIC_DIR / "config.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ui missing")
    return FileResponse(path)


@app.get("/ui/voices")
def ui_voices_get():
    return _voices_ui_payload()


@app.put("/ui/voices")
async def ui_voices_put(request: Request):
    if not voices_file_writable(VOICES_PATH):
        raise HTTPException(status_code=403, detail="voices.json is not writable")
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="voices.json must be a JSON object")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="voices.json must be a JSON object")
    try:
        document = validate_voices_document(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        write_voices_document(VOICES_PATH, document)
    except OSError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    def _reload():
        with lock:
            _reload_voice_overlays()
            _refresh_speakers_maps()
            _rebuild_voice_index()

    await asyncio.to_thread(_reload)
    payload = _voices_ui_payload()
    payload["voices"] = public_voice_names(voice_index, voice_overlays, catalog)
    payload["default"] = default_voice
    return payload


def _reload_voices_locked():
    with lock:
        _reload_voice_overlays()
        _refresh_speakers_maps()
        _rebuild_voice_index()


@app.post("/ui/clone-preset")
async def ui_clone_preset(request: Request):
    if not voices_file_writable(VOICES_PATH):
        raise HTTPException(status_code=403, detail="voices.json is not writable")
    form = await request.form()
    alias = _safe_alias(str(form.get("alias") or ""))
    if not alias:
        raise HTTPException(status_code=400, detail="clone requires a public name")
    ref_text = str(form.get("ref_text") or "").strip()
    if not ref_text:
        raise HTTPException(status_code=400, detail="clone requires ref_text")
    model = str(form.get("model") or "").strip()
    upload = form.get("ref_audio")
    raw = await _read_clone_wav(upload)
    CLONES_DIR.mkdir(parents=True, exist_ok=True)
    dest = CLONES_DIR / f"{alias}.wav"
    dest.write_bytes(raw)
    document, _error = load_voices_document(VOICES_PATH)
    voices = dict(document.get("voices") or {})
    entry = {"kind": "voice_clone", "ref_audio": f"clones/{alias}.wav", "ref_text": ref_text}
    if model:
        entry["model"] = model
    voices[alias] = entry
    try:
        document = validate_voices_document({"voices": voices})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        write_voices_document(VOICES_PATH, document)
    except OSError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    await asyncio.to_thread(_reload_voices_locked)
    payload = _voices_ui_payload()
    payload["voices"] = public_voice_names(voice_index, voice_overlays, catalog)
    payload["default"] = default_voice
    return payload


@app.delete("/ui/clone-preset/{alias}")
async def ui_clone_preset_delete(alias: str):
    if not voices_file_writable(VOICES_PATH):
        raise HTTPException(status_code=403, detail="voices.json is not writable")
    safe = _safe_alias(alias)
    if not safe:
        raise HTTPException(status_code=400, detail="clone requires a public name")
    wav = CLONES_DIR / f"{safe}.wav"
    if wav.is_file():
        try:
            wav.unlink()
        except OSError:
            pass
    document, _error = load_voices_document(VOICES_PATH)
    voices = dict(document.get("voices") or {})
    voices.pop(safe, None)
    try:
        document = validate_voices_document({"voices": voices})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        write_voices_document(VOICES_PATH, document)
    except OSError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    await asyncio.to_thread(_reload_voices_locked)
    payload = _voices_ui_payload()
    payload["voices"] = public_voice_names(voice_index, voice_overlays, catalog)
    payload["default"] = default_voice
    return payload


@app.post("/ui/rescan")
def ui_rescan():
    return _rescan_catalog()


@app.get("/health")
def health():
    if not catalog:
        return {"ok": False, "error": ready_error or "loading"}
    payload = {
        "ok": True,
        "voices": public_voice_names(voice_index, voice_overlays, catalog),
        "default": default_voice,
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "attn": ATTN,
        "model": MODEL_NAME,
        "models": [
            {"id": i, "path": str(p), "loaded": i in loaded, "kind": kind_by.get(i, "custom_voice")}
            for i, p in catalog
        ],
        "policy": LOAD_POLICY,
    }
    xpu_name = xpu_device_name(DEVICE)
    if xpu_name:
        payload["xpu_name"] = xpu_name
    return payload


@app.get("/v1/audio/voices")
@app.get("/v1/voices")
def openai_voices(model: str | None = None):
    if model and not is_public_model_request(model, MODEL_NAME) and model not in catalog_ids():
        raise HTTPException(status_code=400, detail=f"unknown model {model!r}")
    names = public_voice_names(voice_index, voice_overlays, catalog)
    return {
        "object": "list",
        "data": [
            {
                "voice_id": n,
                "name": n,
                "kind": overlay_kind(n, voice_overlays) or kind_by.get(voice_index.get(n.lower(), ("", ""))[0], "custom_voice"),
                "instructions": overlay_instructions(n, voice_overlays) or None,
            }
            for n in names
        ],
        "default": default_voice,
    }


@app.get("/v1/models")
def openai_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    log_api_error(request, 422, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    log_api_error(request, exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log_api_error(request, 500, repr(exc))
    return JSONResponse(status_code=500, content={"detail": str(exc)})



CLONE_WAV_MAX = 50 * 1024 * 1024
_WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}


def _safe_alias(name: str) -> str:
    cleaned = "".join(ch for ch in (name or "") if ch.isalnum() or ch in "._-")
    return cleaned[:64]


def _is_wav_upload(filename: str, content_type: str, raw: bytes) -> bool:
    name = (filename or "").lower()
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if name.endswith(".wav") or name.endswith(".wave") or ctype in _WAV_TYPES:
        return True
    return raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"


async def _read_clone_wav(upload) -> bytes:
    filename = getattr(upload, "filename", "") or ""
    content_type = getattr(upload, "content_type", "") or ""
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="clone requires a wav file")
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="clone requires a wav file")
    if len(raw) > CLONE_WAV_MAX:
        raise HTTPException(status_code=400, detail="clone wav must be 50MB or smaller")
    if not _is_wav_upload(filename, content_type, raw):
        raise HTTPException(status_code=400, detail="clone requires a wav file")
    return raw


def _path_under(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except AttributeError:
        try:
            return os.path.commonpath([str(path), str(root)]) == str(root)
        except ValueError:
            return False


def _resolve_ref_wav(rel: str) -> Path | None:
    rel = (rel or "").strip()
    if not rel:
        return None
    candidate = Path(rel)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    root = VOICES_PATH.parent.resolve()
    path = (VOICES_PATH.parent / rel).resolve()
    if not path.is_file() or path.suffix.lower() != ".wav":
        return None
    if not _path_under(path, root):
        return None
    return path


def _unique_base_id() -> str | None:
    bases = [i for i, p in catalog if (kind_by.get(i) or checkpoint_kind(p)) == "base"]
    if len(bases) == 1:
        return bases[0]
    return None


def _model_kind(model_id: str) -> str:
    return kind_by.get(model_id) or checkpoint_kind(_catalog_path(model_id))


def _generate_audio(
    eng,
    model_id: str,
    text: str,
    speaker: str,
    language: str,
    instruct: str | None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
):
    ra = (ref_audio or "").strip()
    rt = (ref_text or "").strip()
    if ra and rt:
        if not hasattr(eng, "generate_voice_clone"):
            raise HTTPException(status_code=503, detail="qwen-tts build has no generate_voice_clone")
        kind = _model_kind(model_id)
        if kind != "base":
            raise HTTPException(status_code=400, detail="voice clone requires a base checkpoint")
        return eng.generate_voice_clone(text=text, language=language, ref_audio=ra, ref_text=rt)
    kind = _model_kind(model_id)
    if kind == "voice_design":
        desc = (instruct or "").strip()
        if not desc:
            raise HTTPException(status_code=400, detail="voice design requires instructions")
        if not hasattr(eng, "generate_voice_design"):
            raise HTTPException(status_code=503, detail="qwen-tts build has no generate_voice_design")
        return eng.generate_voice_design(text=text, language=language, instruct=desc)
    return eng.generate_custom_voice(text=text, speaker=speaker, language=language, instruct=instruct)


def _run_speech(
    pin_model: str | None,
    voice: str,
    instructions: str | None,
    text: str,
    language: str,
    fmt: str,
    ref_audio: str | None,
    ref_text: str | None,
):
    with lock:
        if pin_model:
            mid = pin_model
            eng = _ensure_engine(mid)
            speakers = speakers_by.get(mid, {})
            model_default = default_voice_by.get(mid, default_voice)
            speaker, fell_back, reason = resolve_voice(voice, speakers, model_default)
        else:
            mid, speaker, fell_back, reason = resolve_voice_route(
                voice,
                voice_index,
                default_voice,
                OPENAI_STOCK_VOICES,
                speaker_key,
            )
            if not mid:
                raise HTTPException(status_code=503, detail="no voices")
            eng = _ensure_engine(mid)
        shot_a = (ref_audio or "").strip()
        shot_t = (ref_text or "").strip()
        if bool(shot_a) ^ bool(shot_t):
            raise HTTPException(status_code=400, detail="clone requires ref_audio and ref_text")
        okind = overlay_kind(voice, voice_overlays)
        if shot_a and shot_t:
            ref_audio, ref_text = shot_a, shot_t
        elif okind == "voice_clone":
            rel, rtext = overlay_clone_ref(voice, voice_overlays)
            wav = _resolve_ref_wav(rel)
            if wav is None:
                raise HTTPException(status_code=400, detail="clone preset missing wav")
            ref_audio, ref_text = str(wav), rtext
            if _model_kind(mid) != "base":
                mid2 = _unique_base_id()
                if mid2 is None:
                    raise HTTPException(status_code=400, detail="voice clone requires a base checkpoint")
                mid = mid2
                eng = _ensure_engine(mid)
        else:
            ref_audio, ref_text = None, None
        clone = bool((ref_audio or "").strip() and (ref_text or "").strip())
        used = (voice.strip() or "clone") if clone else public_voice_id(mid, speaker)
        print(
            f"speech public={MODEL_NAME} model={mid} voice={voice!r} -> {used} "
            f"format={fmt} chars={len(text)} fallback={fell_back}",
            flush=True,
        )
        preset = overlay_instructions(voice, voice_overlays)
        instruct = merge_instructions(preset, instructions)
        wavs, sr = _generate_audio(eng, mid, text, speaker, language, instruct, ref_audio, ref_text)
        audio = np.asarray(wavs[0], dtype=np.float32)
    body, media = encode_audio(audio, sr, fmt)
    return mid, used, fell_back, reason, body, media


def _speech_response(mid, used, fell_back, reason, body, media):
    headers = {"X-TTS-Voice-Used": used, "X-TTS-Model": mid}
    if fell_back:
        headers["X-TTS-Fell-Back"] = "1"
        headers["X-TTS-Fell-Back-Reason"] = reason
    return Response(content=body, media_type=media, headers=headers)


def _pin_base_model(requested: str | None) -> str | None:
    if requested and requested in catalog_ids() and _model_kind(requested) == "base":
        return requested
    return _unique_base_id()


async def _openai_speech_multipart(request: Request):
    form = await request.form()
    text = str(form.get("input") or form.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty input")
    voice = str(form.get("voice") or "")
    model = str(form.get("model") or "")
    language = str(form.get("language") or DEFAULT_LANGUAGE)
    fmt = str(form.get("response_format") or "mp3").lower().strip()
    instructions = form.get("instructions")
    instructions = str(instructions) if instructions is not None else None
    ref_text = str(form.get("ref_text") or "").strip()
    upload = form.get("ref_audio")
    if upload is None or not hasattr(upload, "read") or not ref_text:
        raise HTTPException(status_code=400, detail="clone requires ref_audio and ref_text")
    raw = await _read_clone_wav(upload)
    tmp_dir = CLONES_DIR if CLONES_DIR.parent.is_dir() else Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(tmp_dir))
    try:
        tmp.write(raw)
        tmp.close()
        pin = _pin_base_model(model if model in catalog_ids() else None)
        if pin is None:
            raise HTTPException(status_code=400, detail="voice clone requires a base checkpoint")
        mid, used, fell_back, reason, body, media = await asyncio.to_thread(
            _run_speech,
            pin,
            voice,
            instructions,
            text,
            language,
            fmt,
            tmp.name,
            ref_text,
        )
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except OSError:
            pass
    return _speech_response(mid, used, fell_back, reason, body, media)


@app.post("/v1/audio/speech")
async def openai_speech(request: Request):
    if not catalog:
        raise HTTPException(status_code=503, detail=ready_error or "loading")
    ct = (request.headers.get("content-type") or "").lower()
    if ct.startswith("multipart/form-data"):
        return await _openai_speech_multipart(request)
    try:
        req = OpenAISpeechRequest.model_validate(await request.json())
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")
    if req.model and not is_public_model_request(req.model, MODEL_NAME) and req.model not in catalog_ids():
        raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")
    text = (req.input or req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty input")
    fmt = (req.response_format or "mp3").lower().strip()
    language = req.language or DEFAULT_LANGUAGE
    pin_model = req.model if req.model in catalog_ids() and not is_public_model_request(req.model, MODEL_NAME) else None
    try:
        mid, used, fell_back, reason, body, media = await asyncio.to_thread(
            _run_speech,
            pin_model,
            str(req.voice or ""),
            req.instructions,
            text,
            language,
            fmt,
            None,
            None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    headers = {"X-TTS-Voice-Used": used, "X-TTS-Model": mid}
    if fell_back:
        headers["X-TTS-Fell-Back"] = "1"
        headers["X-TTS-Fell-Back-Reason"] = reason
    return Response(content=body, media_type=media, headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
