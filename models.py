"""Checkpoint catalog, load-policy, and public voice-index helpers. Pure — no torch / qwen_tts."""

from __future__ import annotations

import json
import os
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


def checkpoint_kind(path: Path) -> str:
    """tts_model_type from config.json; missing/unreadable → custom_voice."""
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        return "custom_voice"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "custom_voice"
    raw = data.get("tts_model_type") if isinstance(data, dict) else ""
    return (raw or "").strip().lower() or "custom_voice"


def parse_voice_overlays(
    data: object | None, env_speakers: str
) -> list[tuple[str, str, str | None, str]]:
    """Return (alias, speaker, model_or_None, preset) from voices.json + TTS_SPEAKERS."""
    out: list[tuple[str, str, str | None, str]] = []
    if isinstance(data, dict):
        voices = data.get("voices", data)
        if isinstance(voices, dict):
            for name, spec in voices.items():
                if isinstance(spec, str):
                    out.append((str(name), spec, None, ""))
                elif isinstance(spec, dict):
                    speaker = str(spec.get("speaker") or spec.get("name") or name)
                    model = spec.get("model")
                    model_id = str(model).strip() if model else None
                    raw_preset = spec.get("instructions")
                    preset = raw_preset.strip() if isinstance(raw_preset, str) else ""
                    out.append((str(name), speaker, model_id or None, preset))
        elif isinstance(voices, list):
            for name in voices:
                out.append((str(name), str(name), None, ""))
    for name in (env_speakers or "").split(","):
        name = name.strip()
        if name:
            out.append((name, name, None, ""))
    return out


def voices_file_writable(path: Path) -> bool:
    try:
        if path.is_file():
            return os.access(path, os.W_OK)
        return path.parent.is_dir() and os.access(path.parent, os.W_OK)
    except OSError:
        return False


def load_voices_document(path: Path) -> tuple[dict, str | None]:
    if not path.is_file():
        return {"voices": {}}, None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"voices": {}}, str(exc)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"voices": {}}, f"invalid JSON: {exc}"
    if isinstance(decoded, dict) and "voices" in decoded and isinstance(decoded["voices"], dict):
        return decoded, None
    if isinstance(decoded, dict) and "voices" not in decoded:
        return {"voices": decoded}, None
    return {"voices": {}}, "voices.json must be a JSON object"


def validate_voices_document(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("voices.json must be a JSON object")
    voices = data["voices"] if "voices" in data else data
    if not isinstance(voices, dict):
        raise ValueError("voices must be a JSON object")
    normalized: dict[str, str | dict[str, str]] = {}
    for raw_key, spec in voices.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("empty alias")
        if isinstance(spec, str):
            if not spec.strip():
                raise ValueError(f"{key}: speaker must be a non-empty string")
            normalized[key] = spec
            continue
        if isinstance(spec, dict):
            speaker = spec.get("speaker")
            if not isinstance(speaker, str) or not speaker.strip():
                raise ValueError(f"{key}: speaker must be a non-empty string")
            entry: dict[str, str] = {"speaker": speaker}
            model = spec.get("model")
            if model is not None:
                if not isinstance(model, str):
                    raise ValueError(f"{key}: model must be a string")
                if model.strip():
                    entry["model"] = model
            if "instructions" in spec:
                instructions = spec.get("instructions")
                if not isinstance(instructions, str):
                    raise ValueError(f"{key}: instructions must be a string")
                stripped = instructions.strip()
                if stripped:
                    entry["instructions"] = stripped
            normalized[key] = entry
            continue
        raise ValueError(f"{key}: value must be a string or object")
    return {"voices": normalized}


def write_voices_document(path: Path, document: dict) -> None:
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp, path)
    except OSError:
        if tmp.exists():
            tmp.unlink()
        raise


def _owners_for_speaker(index: dict[str, tuple[str, str]], speaker: str) -> list[str]:
    wanted = speaker.strip().lower()
    seen: list[str] = []
    for mid, real in index.values():
        if real.lower() == wanted and mid not in seen:
            seen.append(mid)
    return seen


def resolve_overlay_target(
    speaker: str,
    model_hint: str | None,
    index: dict[str, tuple[str, str]],
    catalog_ids: list[str],
    default_id: str,
) -> tuple[str, str] | None:
    """Map overlay Maps-to to an existing (model_id, real_speaker), or None."""
    del default_id
    key = (speaker or "").strip().lower()
    if not key:
        return None
    if key in index:
        return index[key]
    stripped = speaker.strip()
    if model_hint and model_hint in catalog_ids:
        pub = public_voice_id(model_hint, stripped).lower()
        if pub in index:
            return index[pub]
        owners = _owners_for_speaker(index, stripped)
        if model_hint in owners or not owners:
            if pub in index:
                return index[pub]
        if owners == [model_hint]:
            return (model_hint, stripped)
    owners = _owners_for_speaker(index, stripped)
    if len(owners) == 1:
        return (owners[0], stripped)
    return None


def merge_instructions(preset: str | None, extra: str | None) -> str | None:
    a = (preset or "").strip()
    b = (extra or "").strip()
    if a and b:
        return f"{a} {b}"
    if a:
        return a
    if b:
        return b
    return None


def overlay_instructions(
    name: str, overlays: list[tuple[str, str, str | None, str]]
) -> str:
    key = (name or "").strip().lower()
    preset = ""
    for alias, _speaker, _hint, item_preset in overlays:
        if alias.strip().lower() == key:
            preset = item_preset
    return preset


def build_voice_index(
    catalog: list[tuple[str, Path]],
    overlays: list[tuple[str, str, str | None, str]],
    default_id: str,
) -> dict[str, tuple[str, str]]:
    """Map lowercased public voice id -> (checkpoint id, speaker). Canonical id is `{model}-{speaker}`."""
    index: dict[str, tuple[str, str]] = {}
    ids = [i for i, _ in catalog]
    for model_id, path in catalog:
        names = checkpoint_speakers(path) or [model_id]
        for name in names:
            index[public_voice_id(model_id, name).lower()] = (model_id, name)
    pairs = _unique_pairs(index)
    speaker_counts = _speaker_counts(pairs)
    for mid, name in pairs:
        if mid.lower() == name.lower() and speaker_counts[name.lower()] == 1:
            index.setdefault(name.lower(), (mid, name))
    for alias, speaker, model_hint, _preset in overlays:
        resolved = resolve_overlay_target(speaker, model_hint, index, ids, default_id)
        if resolved is None:
            continue
        mid, real = resolved
        if alias.strip():
            index[alias.strip().lower()] = (mid, real)
    return index


def _unique_pairs(index: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for pair in index.values():
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _speaker_counts(pairs: list[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _mid, speaker in pairs:
        key = speaker.lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _aliases_by_pair(
    index: dict[str, tuple[str, str]],
    overlays: list[tuple[str, str, str | None, str]] | None,
) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    if not overlays:
        return grouped
    for alias, _speaker, _hint, _preset in overlays:
        name = alias.strip()
        if not name:
            continue
        pair = index.get(name.lower())
        if pair is None:
            continue
        names = grouped.setdefault(pair, [])
        if name not in names:
            names.append(name)
    return grouped


def _listed_name_for_pair(
    mid: str,
    speaker: str,
    listed: list[str],
    index: dict[str, tuple[str, str]],
) -> str:
    pair = (mid, speaker)
    for name in listed:
        if name.lower() in index and index[name.lower()] == pair:
            return name
    return public_voice_id(mid, speaker)


def public_voice_names(
    index: dict[str, tuple[str, str]],
    overlays: list[tuple[str, str, str | None, str]] | None = None,
) -> list[str]:
    pairs = _unique_pairs(index)
    speaker_counts = _speaker_counts(pairs)
    aliases = _aliases_by_pair(index, overlays)
    labels: dict[str, str] = {}
    for mid, speaker in pairs:
        names = aliases.get((mid, speaker))
        if names:
            for name in names:
                labels[name.lower()] = name
            continue
        if mid.lower() == speaker.lower() and speaker_counts[speaker.lower()] == 1:
            labels[speaker.lower()] = speaker
            continue
        pub = public_voice_id(mid, speaker)
        labels[pub.lower()] = pub
    return sorted(labels.values(), key=str.lower)


def public_default_voice(
    index: dict[str, tuple[str, str]],
    requested: str,
    catalog_ids: list[str],
    overlays: list[tuple[str, str, str | None, str]] | None = None,
) -> str:
    listed = public_voice_names(index, overlays)
    if not listed:
        return ""
    requested = (requested or "").strip()
    listed_by_key = {name.lower(): name for name in listed}
    if requested:
        key = requested.lower()
        if key in listed_by_key:
            return listed_by_key[key]
        if key in index:
            mid, speaker = index[key]
            return _listed_name_for_pair(mid, speaker, listed, index)
        owners = _owners_for_speaker(index, requested)
        if len(owners) == 1:
            return _listed_name_for_pair(owners[0], requested, listed, index)
    for mid in catalog_ids:
        for name in listed:
            if name.lower() in index and index[name.lower()][0] == mid:
                return name
    return listed[0]


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
