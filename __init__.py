"""
Shim package to ensure editable installs of the vLLM fork work when running
from the repository root (which contains a top-level `vllm/` directory).

When `python` is executed from the repo root, Python may treat this top-level
`vllm/` directory as a namespace package and skip the real package initializer
located in `vllm/vllm/__init__.py`. That prevents the dynamic exports such as
`LLM` and `SamplingParams` from being available via `from vllm import ...`.

This shim forces the real package to load by importing everything from the
inner `vllm` package.
"""

from importlib import import_module

_real_pkg = import_module("vllm.vllm")

# Re-export everything defined by the real package (including __all__)
globals().update(_real_pkg.__dict__)

__all__ = getattr(_real_pkg, "__all__", [])

