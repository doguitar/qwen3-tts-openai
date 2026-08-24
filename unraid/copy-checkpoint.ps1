# Copy the Serling (or multi-speaker) EasyFinetuning checkpoint onto Unraid.
# Run on the Windows training PC. Edit $UnraidShare if your share path differs.
param(
  [string]$Checkpoint = "F:\qwen3-tts-finetune\output\serling\checkpoint-epoch-2",
  [string]$UnraidShare = "\\tower\appdata\qwen3-tts-openai\models"
)

if (-not (Test-Path $Checkpoint)) {
  throw "Checkpoint not found: $Checkpoint"
}

New-Item -ItemType Directory -Force -Path $UnraidShare | Out-Null

# Skip training_state.pt (~5.5 GB) — inference only needs model.safetensors + tokenizer.
robocopy $Checkpoint $UnraidShare /E /XF training_state.pt /NFL /NDL /NJH
$code = $LASTEXITCODE
if ($code -ge 8) {
  throw "robocopy failed with exit $code"
}

$weights = Join-Path $UnraidShare "model.safetensors"
$tok = Join-Path $UnraidShare "speech_tokenizer\model.safetensors"
if (-not (Test-Path $weights)) { throw "missing $weights" }
if (-not (Test-Path $tok)) {
  Write-Warning "missing $tok — run unraid/download-models.sh on Unraid, or copy from F:\qwen3-tts-finetune\models\Qwen\Qwen3-TTS-12Hz-0.6B-Base\speech_tokenizer"
}

Write-Host "Copied checkpoint to $UnraidShare"
Get-ChildItem $UnraidShare | Select-Object Name, Length
