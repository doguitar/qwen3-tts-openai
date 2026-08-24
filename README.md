# qwen3-tts-openai

Docker image that serves one multi-speaker [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) fine-tune behind an OpenAI-compatible API.

Image: `ghcr.io/doguitar/qwen3-tts-openai`

Mount a local EasyFinetuning checkpoint at `/models`. The `voice` field in `/v1/audio/speech` selects a speaker from that checkpoint.

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Status, loaded speakers, device |
| `GET` | `/v1/models` | Model id (default `tts-1`) |
| `GET` | `/v1/voices` | Speakers from the checkpoint |
| `POST` | `/v1/audio/speech` | Audio body |

```bash
curl http://HOST:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","voice":"SPEAKER","input":"Hello."}' \
  --output out.mp3
```

Optional JSON fields: `instructions`, `language`, `response_format` (`mp3`, `wav`, `pcm`, `opus`, `aac`, `flac`). OpenAI stock voice names fall back to `TTS_DEFAULT_VOICE`.

API errors log the request body.

## Checkpoint layout

Mount one fine-tune directory as `/models`:

```text
/models/config.json
/models/model.safetensors
/models/speech_tokenizer/model.safetensors
/models/tokenizer_config.json
/models/vocab.json
/models/merges.txt
```

If `speech_tokenizer/model.safetensors` is missing from the checkpoint, copy it from the matching Base model (`Qwen/Qwen3-TTS-12Hz-0.6B-Base` or `1.7B-Base`). Skip `training_state.pt` for inference.

Speakers come from the checkpoint. Override with `TTS_SPEAKERS` or `/config/voices.json`:

```json
{
  "voices": {
    "alice": "alice",
    "bob": "bob"
  }
}
```

## Run

```bash
docker run -d --name qwen3-tts-openai \
  --gpus all \
  -p 8080:8080 \
  -e TTS_DEFAULT_VOICE=alice \
  -e TTS_SPEAKERS=alice,bob \
  -v /path/to/checkpoint:/models:ro \
  -v /path/to/config:/config:ro \
  ghcr.io/doguitar/qwen3-tts-openai:latest
```

CPU:

```bash
docker run -d --name qwen3-tts-openai \
  -p 8080:8080 \
  -e TTS_DEVICE=cpu \
  -v /path/to/checkpoint:/models:ro \
  ghcr.io/doguitar/qwen3-tts-openai:latest
```

Private GHCR packages need `docker login ghcr.io`. The package can be set public in GitHub: **Packages → qwen3-tts-openai → Package settings**.

## Unraid

Template: [`unraid/qwen3-tts-openai.xml`](unraid/qwen3-tts-openai.xml)

Copy it to `/boot/config/plugins/dockerMan/templates-user/` and add the container from the Docker tab. Extra params include `--runtime=nvidia --gpus all`.

Host paths:

- `/mnt/user/appdata/qwen3-tts-openai/models` → `/models` (checkpoint)
- `/mnt/user/appdata/qwen3-tts-openai/config` → `/config` (optional `voices.json`)

`unraid/download-models.sh` pulls the image and downloads Base tokenizer files. Copy your own fine-tune over `/models` afterward.

OpenAI-compatible clients: `http://HOST:PORT/v1`, model `tts-1`, `voice` = a speaker name in the checkpoint.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `TTS_MODEL` | `/models` | Fine-tune checkpoint directory |
| `TTS_DEVICE` | `cuda:0` if GPU else `cpu` | Inference device |
| `TTS_SPEAKERS` | *(from checkpoint)* | Comma-separated speaker names |
| `TTS_DEFAULT_VOICE` | first speaker | Empty or unknown `voice` in the request |
| `TTS_VOICES` | `/config/voices.json` | Optional name → speaker map |
| `TTS_LANGUAGE` | `English` | Default synthesis language |
| `TTS_PORT` | `8080` | Listen port |
| `TTS_TOKENIZER` | unset | Extra path to `speech_tokenizer/model.safetensors` |
| `TTS_LOG_BODY_LIMIT` | `8000` | Max request-body chars logged on error |

## Build

`qwen-tts==0.1.1` requires `transformers==4.57.3` and OS `sox`. The image is Ubuntu 22.04 with Torch 2.5.1 cu124 (CUDA libraries come from the torch wheels). Gradio is not installed.

```bash
docker build -t qwen3-tts-openai:local .
```
