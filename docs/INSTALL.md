# Installation and execution environment

Target: Python 3.10, PyTorch 2.4.0, MONAI 1.3.2, NumPy 1.26.4, CUDA-enabled Linux. The release preparation machine is Windows; native Windows selective-scan execution was not established.

Follow the commands in the root README. A compatible NVIDIA driver, CUDA toolkit with `nvcc`, compiler and sufficient GPU memory are needed when building the pinned Mamba/causal-conv1d packages. Install PyTorch first; build the extensions against that environment. If a matching binary wheel is unavailable, compilation is required. The target dependency set is pinned for review, not claimed to be a recovered historical lockfile.

Official references: [PyTorch 2.4 installation commands](https://pytorch.org/get-started/previous-versions/), [Mamba implementation and build guidance](https://github.com/state-spaces/mamba), [MONAI 1.3.2](https://docs.monai.io/en/1.3.2/).

```bash
python -c "import torch; from mamba_ssm import Mamba; print(torch.__version__, torch.cuda.is_available())"
python Code/smoke_cuda.py --classes 8
python Code/smoke_cuda.py --classes 8 --backward
```

The backward check uses batch 2 because the model contains BatchNorm over pooled channel descriptors. Full 128³ training may require substantial GPU memory. Do not reduce the patch size without changing the fixed scan geometry and revalidating the architecture. No CPU replacement for Mamba is installed by production code.

CLI `--help`, manifest syntax checks and CPU IO/metric tests do not require CUDA. Model execution does. The CPU tests install their sequence substitute only inside test subprocesses.
