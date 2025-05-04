from typing import List, Type, Callable
from dataclasses import dataclass
import logging
import ctypes

import ggml

from gglm.utils import Tensor

@dataclass
class GGMLFunction:
    arg_types: List[Type]
    func: Callable

def build_result(name: str, ptr: ggml.ggml_tensor_p):
    tensor = ptr.contents
    shape = [tensor.ne[0], tensor.ne[1], tensor.ne[2], tensor.ne[3]]
    ggml.ggml_set_name(ptr, name.encode())
    return Tensor.from_tensor_ptr(name, ptr)

def get_rows(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return build_result(result_name, ggml.ggml_get_rows(ctx0, a.ptr, b.ptr))

def mul_mat(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    if not (a.shape[0] == b.shape[0] and b.shape[2] % a.shape[2] == 0 and b.shape[3] % a.shape[3] == 0):
        logging.warning(f"Incompatible shapes for matrix multiplication: {a.shape} and {b.shape}")
    
    return build_result(result_name, ggml.ggml_mul_mat(ctx0, a.ptr, b.ptr))

def rms_norm(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor, eps: float):
    return build_result(result_name, ggml.ggml_rms_norm(ctx0, a.ptr, eps))

def reshape_3d(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor, x: int, y: int, z: int):
    return build_result(result_name, ggml.ggml_reshape_3d(ctx0, a.ptr, x, y, z))

def mul(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return build_result(result_name, ggml.ggml_mul(ctx0, a.ptr, b.ptr))

def add(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return build_result(result_name, ggml.ggml_add(ctx0, a.ptr, b.ptr))

def div(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return build_result(result_name, ggml.ggml_div(ctx0, a.ptr, b.ptr))

def rope(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor, b: Tensor, n_rot: int, mode: int, n_ctx: int, n_orig_ctx: int, freq_base: float, freq_scale: float, ext_factor: float, attn_factor: float, beta_fast: float, beta_slow: float):
    return build_result(result_name, ggml.ggml_rope_custom(ctx0, a.ptr, b.ptr, n_rot, mode, n_ctx, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow))

def permute(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor, axis0: int, axis1: int, axis2: int, axis3: int):
    return build_result(result_name, ggml.ggml_permute(ctx0, a.ptr, axis0, axis1, axis2, axis3))

def soft_max(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor, mask: Tensor, scale: float):
    return build_result(result_name, ggml.ggml_soft_max_ext(ctx0, a.ptr, mask.ptr, scale, 0.0))

def cont(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor):
    return build_result(result_name, ggml.ggml_cont(ctx0, a.ptr))

def cont_2d(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor, x: int, y: int):
    return build_result(result_name, ggml.ggml_cont_2d(ctx0, a.ptr, x, y))

def transpose(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor):
    return build_result(result_name, ggml.ggml_transpose(ctx0, a.ptr))

def silu(ctx0: ggml.ggml_context_p, result_name: str, a: Tensor):
    return build_result(result_name, ggml.ggml_silu(ctx0, a.ptr))


GGML_FUNCTIONS = {
    "get_rows": GGMLFunction([Tensor, Tensor], get_rows),
    "mul_mat": GGMLFunction([Tensor, Tensor], mul_mat),
    "rms_norm": GGMLFunction([Tensor, float], rms_norm),
    "reshape_3d": GGMLFunction([Tensor, int, int, int], reshape_3d),
    "mul": GGMLFunction([Tensor, Tensor], mul),
    "add": GGMLFunction([Tensor, Tensor], add),
    "rope": GGMLFunction([Tensor, Tensor, int, int, int, int, float, float, float, float, float, float], rope),
    "permute": GGMLFunction([Tensor, int, int, int, int], permute),
    "soft_max": GGMLFunction([Tensor, Tensor, float], soft_max),
    "cont": GGMLFunction([Tensor], cont),
    "cont_2d": GGMLFunction([Tensor, int, int], cont_2d),
    "transpose": GGMLFunction([Tensor], transpose),
    "silu": GGMLFunction([Tensor], silu),
}