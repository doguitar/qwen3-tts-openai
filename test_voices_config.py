#!/usr/bin/env python3
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from models import (
    load_voices_document,
    validate_voices_document,
    voices_file_writable,
    write_voices_document,
)


class VoicesFileWritableTests(unittest.TestCase):
    def test_missing_file_writable_parent(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "voices.json"
            self.assertTrue(voices_file_writable(path))
            document, error = load_voices_document(path)
            self.assertEqual(document, {"voices": {}})
            self.assertIsNone(error)

    @unittest.skipUnless(os.name != "nt", "POSIX chmod")
    def test_readonly_file(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "voices.json"
            path.write_text('{"voices": {}}', encoding="utf-8")
            os.chmod(path, 0o444)
            try:
                self.assertFalse(voices_file_writable(path))
            finally:
                os.chmod(path, 0o644)


class LoadVoicesDocumentTests(unittest.TestCase):
    def test_valid_wrapped_unchanged(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "voices.json"
            payload = {"voices": {"alice": "alice"}}
            path.write_text(json.dumps(payload), encoding="utf-8")
            document, error = load_voices_document(path)
            self.assertEqual(document, payload)
            self.assertIsNone(error)

    def test_bare_map_wrapped(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "voices.json"
            path.write_text(json.dumps({"alice": "alice"}), encoding="utf-8")
            document, error = load_voices_document(path)
            self.assertEqual(document, {"voices": {"alice": "alice"}})
            self.assertIsNone(error)

    def test_broken_json(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "voices.json"
            path.write_text("{not json", encoding="utf-8")
            document, error = load_voices_document(path)
            self.assertEqual(document, {"voices": {}})
            self.assertIsNotNone(error)
            self.assertIn("invalid JSON", error)


class ValidateVoicesDocumentTests(unittest.TestCase):
    def test_accepts_string_and_object(self):
        out = validate_voices_document(
            {
                "voices": {
                    "alice": "alice",
                    "nick": {"speaker": "bob", "model": "alpha"},
                }
            }
        )
        self.assertEqual(
            out,
            {
                "voices": {
                    "alice": "alice",
                    "nick": {"speaker": "bob", "model": "alpha"},
                }
            },
        )

    def test_rejects_list_value(self):
        with self.assertRaises(ValueError) as ctx:
            validate_voices_document({"alice": ["x"]})
        self.assertIn("alice", str(ctx.exception))

    def test_rejects_empty_alias(self):
        with self.assertRaises(ValueError) as ctx:
            validate_voices_document({"": "alice"})
        self.assertIn("empty alias", str(ctx.exception))

    def test_strips_instructions(self):
        out = validate_voices_document(
            {"voices": {"n": {"speaker": "alice", "instructions": "  Male 40s  "}}}
        )
        self.assertEqual(out["voices"]["n"]["instructions"], "Male 40s")

    def test_accepts_public_id_string(self):
        out = validate_voices_document({"voices": {"mustaine": "mustaine-mustaine"}})
        self.assertEqual(out["voices"]["mustaine"], "mustaine-mustaine")

    def test_rejects_non_string_instructions(self):
        with self.assertRaises(ValueError) as ctx:
            validate_voices_document(
                {"voices": {"n": {"speaker": "alice", "instructions": 1}}}
            )
        self.assertIn("instructions must be a string", str(ctx.exception))

    def test_omits_blank_instructions(self):
        out = validate_voices_document(
            {"voices": {"n": {"speaker": "alice", "instructions": "  "}}}
        )
        self.assertNotIn("instructions", out["voices"]["n"])


class WriteVoicesDocumentTests(unittest.TestCase):
    def test_round_trip(self):
        payload = {
            "voices": {
                "alice": "alice",
                "nick": {"speaker": "bob", "model": "alpha"},
                "narrator": {
                    "speaker": "alpha-alice",
                    "instructions": "Male 40s",
                },
            }
        }
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "voices.json"
            write_voices_document(path, payload)
            document, error = load_voices_document(path)
            self.assertIsNone(error)
            self.assertEqual(document, payload)


if __name__ == "__main__":
    sys.exit(unittest.main())
