#!/bin/bash
# Run on the Unraid host (SSH or User Scripts).
# Pulls the GHCR image, writes config, and downloads the 0.6B Base
# speech tokenizer. Your SFT checkpoint is not on Hugging Face — copy it
# from the training PC after this script (see copy-checkpoint.ps1).
set -euo pipefail

APPDATA="${APPDATA:-/mnt/user/appdata/qwen3-tts-openai}"
MODELS="${APPDATA}/models"
CONFIG="${APPDATA}/config"
IMAGE="${IMAGE:-ghcr.io/doguitar/qwen3-tts-openai:latest}"  # CPU. NVIDIA: IMAGE=...:cuda  Intel Arc: IMAGE=...:xpu
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
GHCR_USER="${GHCR_USER:-doguitar}"

echo "==> appdata ${APPDATA}"
mkdir -p "${MODELS}/speech_tokenizer" "${CONFIG}"

if [[ ! -f "${CONFIG}/voices.json" ]]; then
  cat > "${CONFIG}/voices.json" <<'JSON'
{
  "voices": {
    "serling": "serling",
    "mark": "mark",
    "sinatra": "sinatra"
  }
}
JSON
  echo "==> wrote ${CONFIG}/voices.json"
fi

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  echo "==> docker login ghcr.io as ${GHCR_USER}"
  echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin
else
  echo "==> GITHUB_TOKEN unset; pull will fail if the GHCR package is private"
fi

echo "==> docker pull ${IMAGE}"
docker pull "${IMAGE}"

echo "==> download ${HF_MODEL} speech_tokenizer into ${MODELS}"
docker run --rm \
  -v "${MODELS}:/out" \
  python:3.12-slim \
  bash -c "pip install -q huggingface_hub && huggingface-cli download '${HF_MODEL}' --include 'speech_tokenizer/*' --include 'tokenizer_config.json' --include 'vocab.json' --include 'merges.txt' --include 'config.json' --include 'generation_config.json' --include 'preprocessor_config.json' --local-dir /out"

echo
echo "Tokenizer / base files are in ${MODELS}"
echo "Next: copy each fine-tune into ${MODELS}/<id>/ (example ${MODELS}/serling/)."
echo "Leave ${MODELS}/speech_tokenizer in place. A flat copy onto ${MODELS} still works."
echo "  Windows: unraid/copy-checkpoint.ps1 -ModelId serling"
echo "  Unraid:  rsync -av --exclude training_state.pt /path/to/checkpoint-epoch-2/ ${MODELS}/serling/"
echo
echo "Required after copy:"
echo "  ${MODELS}/<id>/model.safetensors   (or ${MODELS}/model.safetensors for a flat checkpoint)"
echo "  ${MODELS}/speech_tokenizer/model.safetensors"
echo
echo "Then add the Unraid template:"
echo "  curl -fsSL https://raw.githubusercontent.com/doguitar/qwen3-tts-openai/main/unraid/qwen3-tts-openai.xml \\"
echo "    -o /boot/config/plugins/dockerMan/templates-user/my-qwen3-tts-openai.xml"
echo "Docker tab -> Add Container -> Template: qwen3-tts-openai"
echo "Subwave: http://UNRAID-LAN-IP:8080/v1  model tts-1  voice serling"
