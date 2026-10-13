"""Lazy registry. External model code is imported only when its builder is used."""
from importlib import import_module
import re

from .common.adapter import ComparisonModelBundle

_REGISTRY = {}
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def register_comparison_model(name, builder_path):
    """Register 'package.module:function' after a model is actually integrated."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError("model name must be lower_snake_case")
    if name in _REGISTRY:
        raise ValueError("comparison model already registered: " + name)
    if not isinstance(builder_path, str):
        raise TypeError("builder_path must be package.module:function")
    module, separator, function = builder_path.partition(":")
    if not separator or not module or not function or not function.isidentifier():
        raise ValueError("builder_path must be package.module:function")
    _REGISTRY[name] = builder_path


def registered_models():
    return tuple(sorted(_REGISTRY))


def build_comparison_model(name, config):
    """Return model plus adapter; no external models are registered yet."""
    if name not in _REGISTRY:
        raise LookupError(
            f"{name!r} is not integrated; registered: {registered_models()}"
        )
    module_name, function_name = _REGISTRY[name].split(":", 1)
    builder = getattr(import_module(module_name), function_name)
    bundle = builder(config)
    if not isinstance(bundle, ComparisonModelBundle):
        raise TypeError("builder must return ComparisonModelBundle")
    return bundle
