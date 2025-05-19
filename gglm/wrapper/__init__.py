
from typing import TYPE_CHECKING
import ctypes
from gglm.wrapper import gen

ggml_tensor = gen.struct_ggml_tensor
ggml_context = gen.struct_ggml_context
ggml_cgraph = gen.struct_ggml_cgraph
ggml_init_params = gen.struct_ggml_init_params
if TYPE_CHECKING:
    ggml_tensor_p = ctypes._Pointer[ggml_tensor]
    ggml_context_p = ctypes._Pointer[ggml_context]
    ggml_cgraph_p = ctypes._Pointer[ggml_cgraph]
    ggml_init_params_p = ctypes._Pointer[ggml_init_params]
else:
    ggml_tensor_p = ctypes.POINTER(ggml_tensor)
    ggml_context_p = ctypes.POINTER(ggml_context)
    ggml_cgraph_p = ctypes.POINTER(ggml_cgraph)
    ggml_init_params_p = ctypes.POINTER(ggml_init_params)

"""
clang2py src/ggml.c src/ggml-backend.cpp include/ggml.h include/ggml-alloc.h include/ggml-cuda.h include/ggml-cpu.h include/ggml-backend.h \
    --clang-args='-I./include/ -I/usr/include/clang/14' --kind efstu \
    -l /mnt/f/llm-workspace/llama.cpp/build/bin/libggml.so \
    -l /mnt/f/llm-workspace/llama.cpp/build/bin/libggml-base.so \
    -l /mnt/f/llm-workspace/llama.cpp/build/bin/libggml-cpu.so \
    -o /mnt/d/dev/ggml-py-inference/gglm/wrapper/gen.py 
"""
