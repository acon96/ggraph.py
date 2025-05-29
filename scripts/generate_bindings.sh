#!/bin/bash

pushd vendor/llama.cpp/ggml

clang2py src/ggml.c src/ggml-backend.cpp include/ggml.h include/ggml-alloc.h include/ggml-cuda.h include/ggml-cpu.h include/ggml-backend.h src/ggml-backend-impl.h \
    --clang-args='-I./include/ -I/usr/include/clang/14' --kind efstu \
    -l ../build/bin/libggml.so \
    -l ../build/bin/libggml-base.so \
    -l ../build/bin/libggml-cpu.so \
    -l ../build/bin/libllama.so \
    -o ../../../ggraph/wrapper/gen.py 
    # -l ../build/bin/libggml-cuda.so \

popd