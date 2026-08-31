#!/usr/bin/env python3
import unittest

from device import device_kind, inference_settings, select_device


class SelectDeviceTests(unittest.TestCase):
    def test_explicit_request_wins(self):
        self.assertEqual(select_device("xpu", cuda=True, xpu=True), "xpu")
        self.assertEqual(select_device("cpu", cuda=True, xpu=True), "cpu")
        self.assertEqual(select_device("xpu:0", cuda=False, xpu=False), "xpu:0")

    def test_cuda_before_xpu(self):
        self.assertEqual(select_device("", cuda=True, xpu=True), "cuda:0")

    def test_xpu_when_no_cuda(self):
        self.assertEqual(select_device("", cuda=False, xpu=True), "xpu")

    def test_cpu_fallback(self):
        self.assertEqual(select_device("", cuda=False, xpu=False), "cpu")

    def test_whitespace_request_ignored(self):
        self.assertEqual(select_device("  ", cuda=False, xpu=True), "xpu")


class InferenceSettingsTests(unittest.TestCase):
    def test_cuda_bf16_sdpa(self):
        self.assertEqual(inference_settings("cuda:0"), ("bfloat16", "sdpa"))

    def test_xpu_fp16_sdpa_for_alchemist(self):
        self.assertEqual(inference_settings("xpu"), ("float16", "sdpa"))
        self.assertEqual(inference_settings("xpu:0"), ("float16", "sdpa"))

    def test_cpu_fp32_eager(self):
        self.assertEqual(inference_settings("cpu"), ("float32", "eager"))

    def test_dtype_override(self):
        self.assertEqual(inference_settings("xpu", "bfloat16"), ("bfloat16", "sdpa"))
        self.assertEqual(inference_settings("cuda:0", "float16"), ("float16", "sdpa"))

    def test_device_kind(self):
        self.assertEqual(device_kind("xpu:0"), "xpu")
        self.assertEqual(device_kind("cuda:1"), "cuda")
        self.assertEqual(device_kind(""), "cpu")


if __name__ == "__main__":
    unittest.main()
