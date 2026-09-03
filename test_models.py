#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from models import (
    build_voice_index,
    checkpoint_speakers,
    default_model_id,
    discover_checkpoints,
    is_public_model_request,
    parse_load_policy,
    parse_voice_overlays,
    public_default_voice,
    public_voice_id,
    public_voice_names,
    resolve_model_id,
    resolve_voice_route,
)


def _touch_ckpt(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"")


def _write_spk_config(directory: Path, speakers: dict[str, int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"talker_config": {"spk_id": speakers}}),
        encoding="utf-8",
    )
    (directory / "model.safetensors").write_bytes(b"")


class DiscoverCheckpointsTests(unittest.TestCase):
    def test_flat_root_pth(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.pth").write_bytes(b"")
            found = discover_checkpoints(root, "tts-1")
            self.assertEqual([(i, p) for i, p in found], [("tts-1", root)])

    def test_flat_root_ignores_valid_child(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _touch_ckpt(root)
            _touch_ckpt(root / "child-ckpt")
            found = discover_checkpoints(root, "tts-1")
            self.assertEqual([i for i, _ in found], ["tts-1"])
            self.assertEqual(found[0][1], root)

    def test_nested_sorted_skips_tokenizer_and_hidden(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _touch_ckpt(root / "b-ckpt")
            _touch_ckpt(root / "a-ckpt")
            _touch_ckpt(root / "speech_tokenizer")
            _touch_ckpt(root / ".hidden")
            found = discover_checkpoints(root, "tts-1")
            self.assertEqual([i for i, _ in found], ["a-ckpt", "b-ckpt"])

    def test_child_config_only_skipped(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            only_cfg = root / "no-weights"
            only_cfg.mkdir()
            (only_cfg / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(discover_checkpoints(root, "tts-1"), [])


class ParseLoadPolicyTests(unittest.TestCase):
    def test_empty_and_whitespace_default_lazy(self):
        self.assertEqual(parse_load_policy(""), "lazy")
        self.assertEqual(parse_load_policy(" Lazy "), "lazy")

    def test_allowed(self):
        self.assertEqual(parse_load_policy("one"), "one")
        self.assertEqual(parse_load_policy("all"), "all")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            parse_load_policy("resident")
        self.assertIn("lazy", str(ctx.exception))
        self.assertIn("one", str(ctx.exception))
        self.assertIn("all", str(ctx.exception))


class DefaultAndResolveTests(unittest.TestCase):
    def test_default_model_id(self):
        self.assertEqual(default_model_id(["b", "a"], ""), "b")
        self.assertEqual(default_model_id(["a-ckpt", "b-ckpt"], "b-ckpt"), "b-ckpt")
        self.assertEqual(default_model_id(["a-ckpt", "b-ckpt"], "missing"), "a-ckpt")

    def test_resolve_model_id(self):
        ids = ["serling"]
        self.assertEqual(resolve_model_id("", ids, "serling"), "serling")
        self.assertEqual(resolve_model_id("tts-1", ids, "serling"), "serling")
        self.assertEqual(resolve_model_id("qwen3-tts", ids, "serling"), "serling")
        self.assertEqual(resolve_model_id("serling", ids, "serling"), "serling")
        self.assertIsNone(resolve_model_id("nope", ids, "serling"))
        self.assertTrue(is_public_model_request("", "tts-1"))
        self.assertTrue(is_public_model_request("tts-1", "tts-1"))
        self.assertTrue(is_public_model_request("qwen3-tts", "tts-1"))
        self.assertFalse(is_public_model_request("mustaine", "tts-1"))


class VoiceIndexTests(unittest.TestCase):
    def test_checkpoint_speakers_from_spk_id(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_spk_config(root, {"mustaine": 3000, "serling": 3001})
            self.assertEqual(checkpoint_speakers(root), ["mustaine", "serling"])

    def test_checkpoint_speakers_missing_or_invalid(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            root.mkdir(exist_ok=True)
            self.assertEqual(checkpoint_speakers(root), [])
            (root / "config.json").write_text("{", encoding="utf-8")
            self.assertEqual(checkpoint_speakers(root), [])

    def test_build_index_two_solos(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_spk_config(root / "mustaine", {"mustaine": 3000})
            _write_spk_config(root / "serling", {"serling": 3000})
            catalog = discover_checkpoints(root, "tts-1")
            index = build_voice_index(catalog, [], "mustaine")
            self.assertEqual(index["mustaine-mustaine"], ("mustaine", "mustaine"))
            self.assertEqual(index["serling-serling"], ("serling", "serling"))
            self.assertEqual(public_voice_names(index), ["mustaine-mustaine", "serling-serling"])
            self.assertEqual(
                public_default_voice(index, "", ["mustaine", "serling"]),
                "mustaine-mustaine",
            )
            self.assertEqual(
                public_default_voice(index, "serling", ["mustaine", "serling"]),
                "serling-serling",
            )
            self.assertEqual(
                public_default_voice(index, "serling-serling", ["mustaine", "serling"]),
                "serling-serling",
            )

    def test_prefixed_names_keep_colliding_speakers(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_spk_config(root / "cast", {"mustaine": 3003, "serling": 3000})
            _write_spk_config(root / "mustaine", {"mustaine": 3000})
            catalog = discover_checkpoints(root, "tts-1")
            index = build_voice_index(catalog, [], "cast")
            self.assertEqual(index["mustaine-mustaine"], ("mustaine", "mustaine"))
            self.assertEqual(index["cast-mustaine"], ("cast", "mustaine"))
            self.assertEqual(index["cast-serling"], ("cast", "serling"))
            self.assertEqual(
                public_voice_names(index),
                ["cast-mustaine", "cast-serling", "mustaine-mustaine"],
            )

    def test_overlay_alias_and_explicit_model(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_spk_config(root / "mustaine", {"mustaine": 3000})
            catalog = [("mustaine", root / "mustaine")]
            overlays = parse_voice_overlays(
                {"voices": {"dave": {"speaker": "mustaine", "model": "mustaine"}}},
                "mustaine",
            )
            index = build_voice_index(catalog, overlays, "mustaine")
            self.assertEqual(index["dave"], ("mustaine", "mustaine"))
            self.assertEqual(index["mustaine-mustaine"], ("mustaine", "mustaine"))
            self.assertEqual(public_voice_names(index), ["mustaine-mustaine"])

    def test_resolve_voice_route(self):
        index = {
            "mustaine-mustaine": ("mustaine", "mustaine"),
            "serling-serling": ("serling", "serling"),
        }

        def key(value: str) -> str:
            return value.lower().replace("-", "")

        mid, speaker, fell, _reason = resolve_voice_route(
            "mustaine-mustaine", index, "serling-serling", frozenset({"alloy"}), key
        )
        self.assertEqual((mid, speaker, fell), ("mustaine", "mustaine", False))
        mid, speaker, fell, _reason = resolve_voice_route(
            "Mustaine", index, "serling-serling", frozenset({"alloy"}), key
        )
        self.assertEqual((mid, speaker, fell), ("mustaine", "mustaine", False))
        mid, speaker, fell, reason = resolve_voice_route(
            "alloy", index, "serling-serling", frozenset({"alloy"}), key
        )
        self.assertEqual((mid, speaker, fell), ("serling", "serling", True))
        self.assertIn("openai stock", reason)
        mid, speaker, fell, _reason = resolve_voice_route(
            "nope", index, "serling-serling", frozenset(), key
        )
        self.assertEqual((mid, speaker, fell), ("serling", "serling", True))

    def test_bare_speaker_ambiguous_falls_back(self):
        index = {
            "cast-mustaine": ("cast", "mustaine"),
            "mustaine-mustaine": ("mustaine", "mustaine"),
        }

        def key(value: str) -> str:
            return value.lower().replace("-", "")

        mid, speaker, fell, reason = resolve_voice_route(
            "mustaine", index, "cast-mustaine", frozenset(), key
        )
        self.assertEqual((mid, speaker, fell), ("cast", "mustaine", True))
        self.assertIn("unknown voice", reason)
        mid, speaker, fell, _reason = resolve_voice_route(
            "mustaine-mustaine", index, "cast-mustaine", frozenset(), key
        )
        self.assertEqual((mid, speaker, fell), ("mustaine", "mustaine", False))

    def test_public_voice_id(self):
        self.assertEqual(public_voice_id("mustaine", "mustaine"), "mustaine-mustaine")
        self.assertEqual(public_voice_id("cast", "serling"), "cast-serling")


if __name__ == "__main__":
    unittest.main()
