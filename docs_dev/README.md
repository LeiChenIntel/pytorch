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

# Download and install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA 12.2 toolkit (compiler only, no driver — driver should be installed)
# cuda-toolkit version needs to match with the device
sudo apt-get install -y cuda-toolkit-12-2

# Add CUDA to PATH (~/.bashrc)
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version # should show 12.2

# Do not enable virtual environment
cd ~/develop/pytorch
rm -rf build/   # remove old USE_CUDA=OFF build
export TORCH_CUDA_ARCH_LIST="6.1"

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

Pipeline:
The main entrypoint of TorchDynamo.
`_optimize` in `torch/_dynamo/eval_frame.py`
