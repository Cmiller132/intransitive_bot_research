"""Shared test setup: the Triton interpreter on the CPU when no GPU is present,
and the `engine` wheel as a hard requirement."""

import importlib
import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

try:
    importlib.import_module("engine")
except ImportError as error:
    pytest.exit(f"cannot import the engine module ({error}); run `cargo xtask wheel`", returncode=1)
