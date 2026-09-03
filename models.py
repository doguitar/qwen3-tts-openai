"""Checkpoint catalog, load-policy, and public voice-index helpers. Pure — no torch / qwen_tts."""

from __future__ import annotations

import json
from pathlib import Path

WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin", "model.pth", "model.pt")
_ALLOWED_POLICIES = frozenset({"lazy", "one", "all"})
PUBLIC_MODEL_ALIASES = frozenset({"tts-1", "qwen3-tts"})


def is_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    return any((path / name).is_file() for name in WEIGHT_NAMES)


def discover_checkpoints(root: Path, flat_id: str) -> list[tuple[str, Path]]:
    if not root.is_dir():
        return []
    if is_checkpoint(root):
        return [(flat_id, root)]
    found: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name == "speech_tokenizer":
            continue
        if is_checkpoint(child):
            found.append((child.name, child))
    return found


def parse_load_policy(raw: str) -> str:
    value = (raw or "").strip().lower()
    if not value:
        return "lazy"
    if value in _ALLOWED_POLICIES:
        return value
    raise ValueError(f"TTS_LOAD_POLICY must be one of {set(_ALLOWED_POLICIES)}; got {raw!r}")


def default_model_id(catalog_ids: list[str], requested: str) -> str:
    requested = (requested or "").strip()
    if requested and requested in catalog_ids:
        return requested
    return catalog_ids[0]


def resolve_model_id(requested: str, catalog_ids: list[str], default_id: str) -> str | None:
    requested = (requested or "").strip()
    # tts-1 / qwen3-tts always mean the configured default, even if a folder uses that name.
    if not requested or requested in {"tts-1", "qwen3-tts"}:
        return default_id
    if requested in catalog_ids:
        return requested
    return None


def is_public_model_request(requested: str, public_name: str) -> bool:
    requested = (requested or "").strip()
    if not requested:
        return True
    if requested in PUBLIC_MODEL_ALIASES:
        return True
    return requested == (public_name or "").strip()


def public_voice_id(model_id: str, speaker: str) -> str:
    return f"{model_id}-{speaker}"


def checkpoint_speakers(path: Path) -> list[str]:
    """Speaker names from talker_config.spk_id. No weight load."""
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        return []
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    talker = data.get("talker_config") or {}
    spk = talker.get("spk_id")
    if isinstance(spk, dict):
        return [str(name) for name in spk if str(name).strip()]
    if isinstance(spk, list):
        return [str(name) for name in spk if str(name).strip()]
    return []


def parse_voice_overlays(data: object | None, env_speakers: str) -> list[tuple[str, str, str | None]]:
    """Return (alias, speaker, model_or_None) from voices.json + TTS_SPEAKERS."""
    out: list[tuple[str, str, str | None]] = []
    if isinstance(data, dict):
        voices = data.get("voices", data)
        if isinstance(voices, dict):
            for name, spec in voices.items():
                if isinstance(spec, str):
                    out.append((str(name), spec, None))
                elif isinstance(spec, dict):
                    speaker = str(spec.get("speaker") or spec.get("name") or name)
                    model = spec.get("model")
                    model_id = str(model).strip() if model else None
                    out.append((str(name), speaker, model_id or None))
        elif isinstance(voices, list):
            for name in voices:
                out.append((str(name), str(name), None))
    for name in (env_speakers or "").split(","):
        name = name.strip()
        if name:
            out.append((name, name, None))
    return out


def _owners_for_speaker(index: dict[str, tuple[str, str]], speaker: str) -> list[str]:
    wanted = speaker.strip().lower()
    seen: list[str] = []
    for mid, real in index.values():
        if real.lower() == wanted and mid not in seen:
            seen.append(mid)
    return seen


def build_voice_index(
    catalog: list[tuple[str, Path]],
    overlays: list[tuple[str, str, str | None]],
    default_id: str,
) -> dict[str, tuple[str, str]]:
    """Map lowercased public voice id -> (checkpoint id, speaker). Canonical id is `{model}-{speaker}`."""
    index: dict[str, tuple[str, str]] = {}
    ids = [i for i, _ in catalog]
    id_lower = {i.lower(): i for i in ids}
    for model_id, path in catalog:
        names = checkpoint_speakers(path) or [model_id]
        for name in names:
            index[public_voice_id(model_id, name).lower()] = (model_id, name)
    for alias, speaker, model_hint in overlays:
        if model_hint and model_hint in ids:
            mid = model_hint
        else:
            owners = _owners_for_speaker(index, speaker)
            if len(owners) == 1:
                mid = owners[0]
            elif alias.lower() in id_lower:
                mid = id_lower[alias.lower()]
            else:
                mid = default_id
        speaker = speaker.strip() or alias
        pub = public_voice_id(mid, speaker)
        index.setdefault(pub.lower(), (mid, speaker))
        if alias.strip() and alias.lower() != pub.lower():
            index[alias.lower()] = (mid, speaker)
    return index


def public_voice_names(index: dict[str, tuple[str, str]]) -> list[str]:
    labels: dict[str, str] = {}
    for mid, speaker in index.values():
        pub = public_voice_id(mid, speaker)
        labels[pub.lower()] = pub
    return sorted(labels.values(), key=str.lower)


def public_default_voice(
    index: dict[str, tuple[str, str]],
    requested: str,
    catalog_ids: list[str],
) -> str:
    if not index:
        return ""
    requested = (requested or "").strip()
    key = requested.lower()
    if key in index:
        mid, speaker = index[key]
        return public_voice_id(mid, speaker)
    owners = _owners_for_speaker(index, requested)
    if len(owners) == 1:
        return public_voice_id(owners[0], requested)
    for mid in catalog_ids:
        for model_id, speaker in index.values():
            if model_id == mid:
                return public_voice_id(model_id, speaker)
    mid, speaker = next(iter(index.values()))
    return public_voice_id(mid, speaker)


def resolve_voice_route(
    name: str | None,
    index: dict[str, tuple[str, str]],
    default_voice: str,
    stock_voices: set[str] | frozenset[str],
    speaker_key,
) -> tuple[str, str, bool, str]:
    """Return (model_id, speaker, fell_back, reason) for a public `{model}-{voice}` name."""
    if not index:
        return "", default_voice, True, "no voices"
    default_key = (default_voice or "").strip().lower()
    if default_key in index:
        default_mid, default_spk = index[default_key]
    else:
        default_mid, default_spk = next(iter(index.values()))
    key = (name or "").strip().lower()
    if not key or key in stock_voices:
        reason = "empty voice" if not key else f"openai stock voice {key}"
        return default_mid, default_spk, bool(key), reason
    if key in index:
        mid, speaker = index[key]
        return mid, speaker, False, ""
    owners = _owners_for_speaker(index, key)
    if len(owners) == 1:
        return owners[0], key, False, ""
    wanted = speaker_key(key)
    matches: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for alias, (mid, speaker) in index.items():
        pub = public_voice_id(mid, speaker)
        if (
            speaker_key(alias) == wanted
            or speaker_key(pub) == wanted
            or speaker_key(speaker) == wanted
        ):
            pair = (mid, speaker)
            if pair not in seen:
                matches.append(pair)
                seen.add(pair)
    if len(matches) == 1:
        mid, speaker = matches[0]
        return mid, speaker, False, ""
    return default_mid, default_spk, True, f"unknown voice {name!r}; using {default_voice or default_spk}"
