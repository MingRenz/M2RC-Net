# Installation alternatives and compatibility

See the [English README](../README.md#installation) or [中文说明](../README_Zh.md#环境安装) for environment setup. This guide covers binary compatibility and alternative installation routes.

## Binary compatibility

The linked wheels target CPython 3.8 (`cp38`), Linux x86-64, CUDA 11.8 (`cu118`), PyTorch 2.0, and `_GLIBCXX_USE_CXX11_ABI=False`. Use Triton 2.0.0 with this PyTorch environment and select wheels with matching binary tags.

Use the precompiled wheels with a compatible CUDA runtime. For source builds, provide a complete CUDA Toolkit with NVCC and headers, together with a compiler supported by that Toolkit. CUDA Toolkit 11.8 matches the environment in the README.

## Official wheel checksums

Checksums for the official wheels linked in the README:

| Package | Official release | Bytes | SHA256 |
|---|---|---:|---|
| mamba-ssm 1.0.1 | [v1.0.1](https://github.com/state-spaces/mamba/releases/tag/v1.0.1) | 146629955 | `f03b2c9ba7ef63b1a718cfbc31be517b053714cec0606b940502dc0435e57e1d` |
| causal-conv1d 1.0.0 | [v1.0.0](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.0.0) | 9145439 | `39d944eb4cfc95061f0cd698faf8f8b435b0bb3be08df75c3361020428f266b8` |

## PyPI option

Both exact versions are available on PyPI: [mamba-ssm 1.0.1](https://pypi.org/project/mamba-ssm/1.0.1/) and [causal-conv1d 1.0.0](https://pypi.org/project/causal-conv1d/1.0.0/). Their PyPI files are source distributions. Their setup code may retrieve an upstream binary wheel or compile locally, depending on the environment.

After installing PyTorch and the dependencies listed in the README:

```bash
python -m pip install --no-build-isolation --no-deps causal-conv1d==1.0.0
python -m pip install --no-build-isolation --no-deps mamba-ssm==1.0.1
```

The package installer retrieves a matching upstream wheel when available and otherwise compiles locally. Provide the CUDA build toolchain for local compilation, or use the matching wheel URLs in the README.

## Explicit official source builds

Official tagged sources:

- [causal-conv1d v1.0.0](https://github.com/Dao-AILab/causal-conv1d/tree/v1.0.0)
- [Mamba v1.0.1](https://github.com/state-spaces/mamba/tree/v1.0.1)

Set CUDA_HOME to a complete CUDA 11.8 Toolkit. Replace the placeholder below with its installation directory:

```bash
export CUDA_HOME=/path/to/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
nvcc --version

CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps \
  "https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.0.0.tar.gz"

MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps \
  "https://github.com/state-spaces/mamba/archive/refs/tags/v1.0.1.tar.gz"
```

The force-build environment variables select local compilation in the corresponding tagged `setup.py` files.
