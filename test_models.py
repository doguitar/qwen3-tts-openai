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
        ids = ["bravo"]
        self.assertEqual(resolve_model_id("", ids, "bravo"), "bravo")
        self.assertEqual(resolve_model_id("tts-1", ids, "bravo"), "bravo")
        self.assertEqual(resolve_model_id("qwen3-tts", ids, "bravo"), "bravo")
        self.assertEqual(resolve_model_id("bravo", ids, "bravo"), "bravo")
        self.assertIsNone(resolve_model_id("nope", ids, "bravo"))
        self.assertTrue(is_public_model_request("", "tts-1"))
        self.assertTrue(is_public_model_request("tts-1", "tts-1"))
        self.assertTrue(is_public_model_request("qwen3-tts", "tts-1"))
        self.assertFalse(is_public_model_request("alpha", "tts-1"))


class VoiceIndexTests(unittest.TestCase):
    def test_checkpoint_speakers_from_spk_id(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_spk_config(root, {"alice": 3000, "bob": 3001})
            self.assertEqual(checkpoint_speakers(root), ["alice", "bob"])

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
            _write_spk_config(root / "alpha", {"alice": 3000})
            _write_spk_config(root / "bravo", {"bob": 3000})
            catalog = discover_checkpoints(root, "tts-1")
            index = build_voice_index(catalog, [], "alpha")
            self.assertEqual(index["alpha-alice"], ("alpha", "alice"))
            self.assertEqual(index["bravo-bob"], ("bravo", "bob"))
            self.assertEqual(public_voice_names(index), ["alpha-alice", "bravo-bob"])
            self.assertEqual(
                public_default_voice(index, "", ["alpha", "bravo"]),
                "alpha-alice",
            )
            self.assertEqual(
                public_default_voice(index, "bob", ["alpha", "bravo"]),
                "bravo-bob",
            )
            self.assertEqual(
                public_default_voice(index, "bravo-bob", ["alpha", "bravo"]),
                "bravo-bob",
            )

    def test_prefixed_names_keep_colliding_speakers(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_spk_config(root / "cast", {"alice": 3003, "bob": 3000})
            _write_spk_config(root / "alpha", {"alice": 3000})
            catalog = discover_checkpoints(root, "tts-1")
            index = build_voice_index(catalog, [], "cast")
            self.assertEqual(index["alpha-alice"], ("alpha", "alice"))
            self.assertEqual(index["cast-alice"], ("cast", "alice"))
            self.assertEqual(index["cast-bob"], ("cast", "bob"))
            self.assertEqual(
                public_voice_names(index),
                ["alpha-alice", "cast-alice", "cast-bob"],
            )

    def test_overlay_alias_and_explicit_model(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_spk_config(root / "alpha", {"alice": 3000})
            catalog = [("alpha", root / "alpha")]
            overlays = parse_voice_overlays(
                {"voices": {"nickname": {"speaker": "alice", "model": "alpha"}}},
                "alice",
            )
            index = build_voice_index(catalog, overlays, "alpha")
            self.assertEqual(index["nickname"], ("alpha", "alice"))
            self.assertEqual(index["alpha-alice"], ("alpha", "alice"))
            self.assertEqual(public_voice_names(index), ["alpha-alice"])

    def test_resolve_voice_route(self):
        index = {
            "alpha-alice": ("alpha", "alice"),
            "bravo-bob": ("bravo", "bob"),
        }

        def key(value: str) -> str:
            return value.lower().replace("-", "")

        mid, speaker, fell, _reason = resolve_voice_route(
            "alpha-alice", index, "bravo-bob", frozenset({"alloy"}), key
        )
        self.assertEqual((mid, speaker, fell), ("alpha", "alice", False))
        mid, speaker, fell, _reason = resolve_voice_route(
            "Alice", index, "bravo-bob", frozenset({"alloy"}), key
        )
        self.assertEqual((mid, speaker, fell), ("alpha", "alice", False))
        mid, speaker, fell, reason = resolve_voice_route(
            "alloy", index, "bravo-bob", frozenset({"alloy"}), key
        )
        self.assertEqual((mid, speaker, fell), ("bravo", "bob", True))
        self.assertIn("openai stock", reason)
        mid, speaker, fell, _reason = resolve_voice_route(
            "nope", index, "bravo-bob", frozenset(), key
        )
        self.assertEqual((mid, speaker, fell), ("bravo", "bob", True))

    def test_bare_speaker_ambiguous_falls_back(self):
        index = {
            "cast-alice": ("cast", "alice"),
            "alpha-alice": ("alpha", "alice"),
        }

        def key(value: str) -> str:
            return value.lower().replace("-", "")

        mid, speaker, fell, reason = resolve_voice_route(
            "alice", index, "cast-alice", frozenset(), key
        )
        self.assertEqual((mid, speaker, fell), ("cast", "alice", True))
        self.assertIn("unknown voice", reason)
        mid, speaker, fell, _reason = resolve_voice_route(
            "alpha-alice", index, "cast-alice", frozenset(), key
        )
        self.assertEqual((mid, speaker, fell), ("alpha", "alice", False))

    def test_public_voice_id(self):
        self.assertEqual(public_voice_id("alpha", "alice"), "alpha-alice")
        self.assertEqual(public_voice_id("cast", "bob"), "cast-bob")


if __name__ == "__main__":
    unittest.main()
