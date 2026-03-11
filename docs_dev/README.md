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

CMake command for reference:
```cmake
  cmake -GNinja -DBUILD_PYTHON=True -DBUILD_TEST=True -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/leichen1/develop/pytorch/torch -DCMAKE_PREFIX_PATH=/home/leichen1/develop/pytorch/pytorch-dev/lib/python3.10/site-packages;/home/leichen1/develop/pytorch/pytorch-dev: -DPython_EXECUTABLE=/home/leichen1/develop/pytorch/pytorch-dev/bin/python -DPython_NumPy_INCLUDE_DIR=/home/leichen1/develop/pytorch/pytorch-dev/lib/python3.10/site-packages/numpy/_core/include -DTORCH_BUILD_VERSION=2.11.0a0+gitf31baaa -DUSE_NUMPY=True -DUSE_ROCM=0 -DUSE_XPU=0 /home/leichen1/develop/pytorch
```

Run test cases:
```bash
python test-0.py
# logs dumps as
# 2.11.0a0+gitf31baaa
# /home/leichen1/develop/pytorch/torch/__init__.py
```
