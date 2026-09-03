# qwen3-tts-openai

Docker image that serves one or more [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) fine-tunes behind an OpenAI-compatible API. Clients see one model (`tts-1`); `voice` selects the speaker and the matching checkpoint.

Images:

| Tag | Backend |
|---|---|
| `:latest` / `:cpu` | CPU (PyTorch CPU wheels, smaller) |
| `:cuda` | NVIDIA CUDA |
| `:xpu` | Intel Arc / XPU |

Mount checkpoints at `/models`. The `voice` field is `{folder}-{speaker}` (for example `alpha-alice`).

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Status, public voices, device |
| `GET` | `/v1/models` | One public id (`TTS_MODEL_NAME`, default `tts-1`) |
| `GET` | `/v1/voices` | `{folder}-{speaker}` for every checkpoint |
| `POST` | `/v1/audio/speech` | Audio body; `voice` is `{folder}-{speaker}` |

```bash
curl http://HOST:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","voice":"alpha-alice","input":"Hello."}' \
  --output out.mp3
```

Optional JSON fields: `instructions`, `language`, `response_format` (`mp3`, `wav`, `pcm`, `opus`, `aac`, `flac`). OpenAI stock voice names fall back to `TTS_DEFAULT_VOICE`.

API errors log the request body.

## Checkpoint layout

Mount one fine-tune directory as `/models`, or a parent of per-model subfolders:

```text
/models/<id>/config.json
/models/<id>/model.safetensors
/models/speech_tokenizer/model.safetensors
```

A flat checkpoint at `/models` (`config.json` + weights) still works. Folder names stay private; `GET /v1/models` always returns `TTS_MODEL_NAME`.

If `speech_tokenizer/model.safetensors` is missing from the checkpoint, copy it from the matching Base model (`Qwen/Qwen3-TTS-12Hz-0.6B-Base` or `1.7B-Base`). Skip `training_state.pt` for inference.

Public voices are `{folder}-{speaker}` from each checkpoint's `talker_config.spk_id`. Optional aliases in `TTS_SPEAKERS` or `/config/voices.json`:

```json
{
  "voices": {
    "alice": "alice",
    "bob": "bob"
  }
}
```

## Run

CPU (`:latest` is this image):

```bash
docker run -d --name qwen3-tts-openai \
  -p 8080:8080 \
  -e TTS_DEVICE=cpu \
  -e TTS_DEFAULT_VOICE=alice \
  -e TTS_SPEAKERS=alice,bob \
  -v /path/to/checkpoint:/models:ro \
  -v /path/to/config:/config:ro \
  ghcr.io/doguitar/qwen3-tts-openai:latest
```

NVIDIA CUDA. Use `:cuda`, not `:latest`. The CPU wheels have no CUDA kernels.

```bash
docker run -d --name qwen3-tts-openai \
  --gpus all \
  -p 8080:8080 \
  -e TTS_DEVICE=cuda:0 \
  -e TTS_DEFAULT_VOICE=alice \
  -e TTS_SPEAKERS=alice,bob \
  -v /path/to/checkpoint:/models:ro \
  -v /path/to/config:/config:ro \
  ghcr.io/doguitar/qwen3-tts-openai:cuda
```

Intel Arc (A380 / Alchemist and later). Use the `:xpu` image. The CUDA and CPU wheels have no `torch.xpu`. Pass the render node; `--gpus all` is NVIDIA-only. A380 is 6GB: use the 0.6B fine-tune. Default dtype is float32; float16 loads then crashes in `torch._assert_async` on torch 2.13+xpu.

```bash
docker run -d --name qwen3-tts-openai \
  --device /dev/dri \
  --group-add $(stat -c '%g' /dev/dri/renderD128) \
  -p 8080:8080 \
  -e TTS_DEVICE=xpu \
  -e TTS_DEFAULT_VOICE=alice \
  -e TTS_SPEAKERS=alice,bob \
  -v /path/to/checkpoint:/models:ro \
  -v /path/to/config:/config:ro \
  ghcr.io/doguitar/qwen3-tts-openai:xpu
```

Host needs a kernel/driver that sees the Arc GPU (`i915` or `xe`) and **Resizable BAR** (Above 4G Decoding + Re-Size BAR in BIOS). A 256MB BAR is not enough: `torch.xpu` may list the GPU, then kernels fail with `could not make an engine with allocator` or SIGSEGV in `libze_intel_gpu`. `lspci -vv` should show BAR 2 at 4GB–8GB, not 256MB. The `:xpu` image must use Intel's Ubuntu **unified** GPU apt channel (`libze-intel-gpu1` 25.18+). The older **client** channel (24.39 on jammy) produces the same allocator error even with an 8GB BAR. In the container, `/health` should report `"device": "xpu"` and an `xpu_name` such as `Intel(R) Arc(TM) A380 Graphics`. Auto-detect order when `TTS_DEVICE` is unset: CUDA, then XPU, then CPU.

Private GHCR packages need `docker login ghcr.io`. The package can be set public in GitHub: **Packages → qwen3-tts-openai → Package settings**.

## Unraid

Copy a template to `/boot/config/plugins/dockerMan/templates-user/` and add the container from the Docker tab.

- CPU: [`unraid/qwen3-tts-openai-cpu.xml`](unraid/qwen3-tts-openai-cpu.xml) (`:cpu`, same as `:latest`)
- NVIDIA: [`unraid/qwen3-tts-openai.xml`](unraid/qwen3-tts-openai.xml) (`:cuda`, `--runtime=nvidia --gpus all`)
- Intel Arc: [`unraid/qwen3-tts-openai-xpu.xml`](unraid/qwen3-tts-openai-xpu.xml) (`:xpu`, `--device=/dev/dri --group-add 18`, `TTS_DEVICE=xpu`). Unraid 7.0+ is the realistic floor for Alchemist. Enable Resizable BAR in BIOS; 256MB BAR 2 will not run XPU inference.

Host paths:

- `/mnt/user/appdata/qwen3-tts-openai/models` → `/models` (checkpoint)
- `/mnt/user/appdata/qwen3-tts-openai/config` → `/config` (optional `voices.json`)

OpenAI-compatible clients: `http://HOST:PORT/v1`, model `tts-1`, `voice` = `{folder}-{speaker}`.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `TTS_MODEL` | `/models` | Fine-tune checkpoint directory, or parent of per-model subfolders |
| `TTS_LOAD_POLICY` | `lazy` | `lazy` (load on first use), `one` (one resident), or `all` |
| `TTS_DEFAULT_MODEL` | first sorted id | Fallback checkpoint / default voice owner |
| `TTS_MODEL_NAME` | `tts-1` | Public model id listed by `GET /v1/models` |
| `TTS_DEVICE` | `cuda:0`, else `xpu`, else `cpu` | Inference device (`cuda:0`, `xpu`, `cpu`) |
| `TTS_DTYPE` | `bfloat16` on CUDA, `float32` on XPU, `float32` on CPU | Override torch dtype |
| `TTS_SPEAKERS` | *(from checkpoint)* | Comma-separated speaker names |
| `TTS_DEFAULT_VOICE` | first speaker | Empty or unknown `voice` in the request |
| `TTS_VOICES` | `/config/voices.json` | Optional name → speaker map |
| `TTS_LANGUAGE` | `English` | Default synthesis language |
| `TTS_PORT` | `8080` | Listen port |
| `TTS_TOKENIZER` | unset | Extra path to `speech_tokenizer/model.safetensors` |
| `TTS_LOG_BODY_LIMIT` | `8000` | Max request-body chars logged on error |

## Build

`qwen-tts==0.1.1` requires `transformers==4.57.3` and OS `sox`. Default image is CPU (Ubuntu 22.04, Torch 2.5.1 CPU wheels). CUDA image: Torch 2.5.1 cu124. XPU image: official PyTorch `whl/xpu` wheels plus Intel Level Zero userspace. Gradio is not installed.

```bash
docker build --build-arg TORCH_BACKEND=cpu -t qwen3-tts-openai:cpu .
docker build --build-arg TORCH_BACKEND=cuda -t qwen3-tts-openai:cuda .
docker build --build-arg TORCH_BACKEND=xpu -t qwen3-tts-openai:xpu .
```
