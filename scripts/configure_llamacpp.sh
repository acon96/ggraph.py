#!/bin/bash

mkdir -p vendor/llama.cpp/build
pushd vendor/llama.cpp/build
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_NATIVE=ON \
    # -DGGML_CUDA=ON

popd
