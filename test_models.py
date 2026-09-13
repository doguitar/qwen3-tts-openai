#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from models import (
    VoiceOverlay,
    build_voice_index,
    checkpoint_kind,
    checkpoint_speakers,
    default_model_id,
    discover_checkpoints,
    is_public_model_request,
    merge_instructions,
    overlay_clone_ref,
    overlay_instructions,
    overlay_kind,
    parse_load_policy,
    parse_voice_overlays,
    public_default_voice,
    public_voice_id,
    public_voice_names,
    resolve_model_id,
    resolve_overlay_target,
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

    def test_unique_matching_folder_and_speaker_short_name(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_spk_config(root / "mustaine", {"mustaine": 3000})
            _write_spk_config(root / "alpha", {"alice": 3000})
            catalog = [
                ("alpha", root / "alpha"),
                ("mustaine", root / "mustaine"),
            ]
            index = build_voice_index(catalog, [], "alpha")
            self.assertEqual(index["mustaine"], ("mustaine", "mustaine"))
            self.assertEqual(index["mustaine-mustaine"], ("mustaine", "mustaine"))
            self.assertEqual(public_voice_names(index), ["alpha-alice", "mustaine"])
            self.assertEqual(
                public_default_voice(index, "", ["alpha", "mustaine"]),
                "alpha-alice",
            )
            self.assertEqual(
                public_default_voice(index, "mustaine-mustaine", ["alpha", "mustaine"]),
                "mustaine",
            )

            _write_spk_config(root / "other", {"mustaine": 3001})
            catalog_clash = catalog + [("other", root / "other")]
            clash = build_voice_index(catalog_clash, [], "alpha")
            self.assertNotIn("mustaine", clash)
            self.assertEqual(
                public_voice_names(clash),
                ["alpha-alice", "mustaine-mustaine", "other-mustaine"],
            )

    def test_overlay_public_id_short_name(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_spk_config(root / "mustaine", {"mustaine": 3000})
            catalog = [("mustaine", root / "mustaine")]
            overlays = parse_voice_overlays({"voices": {"mustaine": "mustaine-mustaine"}}, "")
            index = build_voice_index(catalog, overlays, "mustaine")
            self.assertEqual(index["mustaine"], ("mustaine", "mustaine"))
            self.assertEqual(index["mustaine-mustaine"], ("mustaine", "mustaine"))
            self.assertEqual(public_voice_names(index), ["mustaine"])
            self.assertEqual(public_voice_names(index, overlays), ["mustaine"])
            self.assertNotIn("mustaine-mustaine-mustaine", index)
            self.assertNotIn("mustaine-mustaine", public_voice_names(index, overlays))

            overlays_obj = parse_voice_overlays(
                {"voices": {"mustaine": {"speaker": "mustaine-mustaine"}}},
                "",
            )
            index_obj = build_voice_index(catalog, overlays_obj, "mustaine")
            self.assertEqual(index_obj["mustaine"], ("mustaine", "mustaine"))
            self.assertEqual(index_obj["mustaine-mustaine"], ("mustaine", "mustaine"))
            self.assertNotIn("mustaine-mustaine-mustaine", index_obj)

            skipped = parse_voice_overlays({"voices": {"short": "nope"}}, "")
            index_skip = build_voice_index(catalog, skipped, "mustaine")
            self.assertNotIn("short", index_skip)

    def test_merge_and_overlay_instructions(self):
        self.assertEqual(merge_instructions("Male 40s", "whisper"), "Male 40s whisper")
        self.assertEqual(merge_instructions("Male 40s", ""), "Male 40s")
        self.assertEqual(merge_instructions("", "whisper"), "whisper")
        self.assertIsNone(merge_instructions("  ", None))
        overlays = [VoiceOverlay("Narrator", "alice", "alpha", "Male 40s", "", "", "")]
        self.assertEqual(overlay_instructions("narrator", overlays), "Male 40s")
        self.assertEqual(overlay_instructions("missing", overlays), "")

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
            self.assertEqual(public_voice_names(index, overlays), ["alice", "nickname"])

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


class CheckpointKindTests(unittest.TestCase):
    def test_missing_config(self):
        with tempfile.TemporaryDirectory() as raw:
            self.assertEqual(checkpoint_kind(Path(raw)), "custom_voice")

    def test_empty_object(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(checkpoint_kind(root), "custom_voice")

    def test_voice_design_normalized(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.json").write_text(
                json.dumps({"tts_model_type": " Voice_Design "}), encoding="utf-8"
            )
            self.assertEqual(checkpoint_kind(root), "voice_design")

    def test_custom_voice(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.json").write_text(
                json.dumps({"tts_model_type": "custom_voice"}), encoding="utf-8"
            )
            self.assertEqual(checkpoint_kind(root), "custom_voice")

    def test_broken_json(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.json").write_text("{", encoding="utf-8")
            self.assertEqual(checkpoint_kind(root), "custom_voice")

    def test_base_kind(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.json").write_text(
                json.dumps({"tts_model_type": "base"}), encoding="utf-8"
            )
            self.assertEqual(checkpoint_kind(root), "base")


class VoiceCloneIndexTests(unittest.TestCase):
    def _base_catalog(self, root: Path, folders: list[str]) -> list[tuple[str, Path]]:
        catalog = []
        for name in folders:
            path = root / name
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text(
                json.dumps({"tts_model_type": "base", "talker_config": {"spk_id": {}}}),
                encoding="utf-8",
            )
            (path / "model.safetensors").write_bytes(b"")
            catalog.append((name, path))
        return catalog

    def test_clone_overlay_unique_base(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            catalog = self._base_catalog(root, ["qbase"])
            overlays = parse_voice_overlays(
                {
                    "voices": {
                        "jane": {
                            "kind": "voice_clone",
                            "ref_audio": "clones/jane.wav",
                            "ref_text": "Hello there.",
                        }
                    }
                },
                "",
            )
            index = build_voice_index(catalog, overlays, "qbase")
            self.assertEqual(index["jane"], ("qbase", "jane"))
            self.assertEqual(public_voice_names(index, overlays), ["jane"])
            self.assertNotIn("qbase-jane", public_voice_names(index, overlays))
            self.assertEqual(overlay_kind("jane", overlays), "voice_clone")
            self.assertEqual(overlay_clone_ref("jane", overlays), ("clones/jane.wav", "Hello there."))

    def test_clone_skipped_without_model_when_two_bases(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            catalog = self._base_catalog(root, ["qbase", "qbase2"])
            overlays = parse_voice_overlays(
                {
                    "voices": {
                        "jane": {
                            "kind": "voice_clone",
                            "ref_audio": "clones/jane.wav",
                            "ref_text": "Hello there.",
                        }
                    }
                },
                "",
            )
            index = build_voice_index(catalog, overlays, "qbase")
            self.assertNotIn("jane", index)

    def test_clone_explicit_model_with_two_bases(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            catalog = self._base_catalog(root, ["qbase", "qbase2"])
            overlays = parse_voice_overlays(
                {
                    "voices": {
                        "jane": {
                            "kind": "voice_clone",
                            "model": "qbase",
                            "ref_audio": "clones/jane.wav",
                            "ref_text": "Hello there.",
                        }
                    }
                },
                "",
            )
            index = build_voice_index(catalog, overlays, "qbase")
            self.assertEqual(index["jane"], ("qbase", "jane"))



if __name__ == "__main__":
    unittest.main()
