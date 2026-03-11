### How to build using virtualenv

```bash
# Clone the repository
git clone https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

# Create a virtual environment
virtualenv pytorch-dev
source pytorch-dev/bin/activate

# Upgrade pip if required
pip install --upgrade pip

# Install dependencies
pip install --group dev
pip install mkl-static mkl-include
pip install ninja

# Build PyTorch
export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}:${CMAKE_PREFIX_PATH}"
MAX_JOBS=16 USE_ROCM=0 USE_XPU=0 python -m pip install --no-build-isolation -v -e .
```
