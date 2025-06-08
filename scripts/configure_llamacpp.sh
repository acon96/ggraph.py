#!/bin/bash

mkdir -p vendor/llama.cpp/build
pushd vendor/llama.cpp/build

# detect cuda version
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found. Skipping CUDA backend."
    CUDA_VERSION="none"
    CUDA_FLAG=OFF
else
    CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //; s/,//')
    CUDA_FLAG=ON
    echo "Configuring for CUDA $CUDA_VERSION."
fi


# cuda requires specific gcc versions
if [[ $CUDA_VERSION == 11.* ]]; then
    echo "Using GCC 10 for CUDA 11.x"
    export CC=$(which gcc-10)
    export CXX=$(which g++-10)
    export CUDAHOSTCXX=$(which g++-10)
elif [[ $CUDA_VERSION == 12.* ]]; then
    echo "Using GCC 12 for CUDA 12.x"
    export CC=$(which gcc-12)
    export CXX=$(which g++-12)
    export CUDAHOSTCXX=$(which g++-12)
fi

cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA=$CUDA_FLAG -DCMAKE_CUDA_ARCHITECTURES=all-major -DCUDA_NVCC_FLAGS=-Wno-deprecated-gpu-targets

popd
