# qwen3-tts-openai

Docker image that serves **one multi-speaker Qwen3-TTS fine-tune** behind an OpenAI-compatible API.

Build note: `qwen-tts==0.1.1` requires **`transformers==4.57.3`** (not 4.57.6) plus OS `sox`. The image is CUDA 12.4 + Torch 2.5.1 cu124.

Image: `ghcr.io/doguitar/qwen3-tts-openai`

This is not llama.cpp and not Faster-Qwen3-TTS-Wyoming. Mount your EasyFinetuning checkpoint (several speakers in one `model.safetensors`) and pick the speaker with the `voice` field.

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | `{"ok": true, "voices": [...]}` |
| `GET` | `/v1/models` | model id `tts-1` |
| `GET` | `/v1/voices` | speakers from the checkpoint |
| `POST` | `/v1/audio/speech` | WAV body |

```bash
curl http://HOST:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","voice":"serling","input":"Submitted for your approval."}' \
  --output out.wav
```

Optional JSON fields: `instructions` (style prompt), `language`, `response_format` (`wav` or `pcm`).

## Train once with many speakers

Use [Qwen3-TTS Easy Finetuning](https://github.com/mozi1924/Qwen3-TTS-EasyFinetuning) multi-speaker train, then mount **that one checkpoint** as `/models`:

```text
/models/config.json
/models/model.safetensors
/models/speech_tokenizer/model.safetensors   # copy from the Base model if missing
/models/tokenizer_config.json
/models/vocab.json
/models/merges.txt
```

Speakers are discovered from the checkpoint. Override with env or `/config/voices.json`:

```json
{
  "voices": {
    "serling": "serling",
    "mark": "mark",
    "sinatra": "sinatra"
  }
}
```

## docker run (Unraid / NVIDIA)

Packages are published by GitHub Actions on push to `main`.

```bash
docker run -d --name qwen3-tts-openai \
  --gpus all \
  -p 8080:8080 \
  -e TTS_DEFAULT_VOICE=serling \
  -e TTS_SPEAKERS=serling,mark,sinatra \
  -v /mnt/user/appdata/qwen3-tts-openai/models:/models:ro \
  -v /mnt/user/appdata/qwen3-tts-openai/config:/config:ro \
  ghcr.io/doguitar/qwen3-tts-openai:latest
```

CPU (slow):

```bash
docker run -d --name qwen3-tts-openai \
  -p 8080:8080 \
  -e TTS_DEVICE=cpu \
  -v /path/to/checkpoint:/models:ro \
  ghcr.io/doguitar/qwen3-tts-openai:latest
```

Pulling a private GHCR package:

```bash
echo YOUR_GITHUB_PAT | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin
```

In the GitHub repo: **Packages → qwen3-tts-openai → Package settings → change visibility to public** if you want unauthenticated pulls.

## Subwave

Cloud / OpenAI-compatible:

- Server URL: `http://UNRAID-OR-GPU-HOST:8080/v1`
- Model: `tts-1`
- Persona voice: `serling` (or any speaker name in the fine-tune)

Do not use `127.0.0.1` from a container on another host.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `TTS_MODEL` | `/models` | Fine-tune checkpoint directory |
| `TTS_DEVICE` | `cuda:0` if GPU else `cpu` | Inference device |
| `TTS_SPEAKERS` | *(from checkpoint)* | Comma-separated speaker names |
| `TTS_DEFAULT_VOICE` | first speaker | Empty `voice` in the request |
| `TTS_VOICES` | `/config/voices.json` | Optional name → speaker map |
| `TTS_LANGUAGE` | `English` | Default synthesis language |
| `TTS_PORT` | `8080` | Listen port |
| `TTS_TOKENIZER` | unset | Extra path to `speech_tokenizer/model.safetensors` |
