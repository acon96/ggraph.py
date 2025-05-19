from typing import List, Type, Callable
from dataclasses import dataclass
import logging
import ctypes

from gglm.wrapper import ggml_context_p, ggml_tensor
from gglm.wrapper import gen
from gglm.utils import Tensor


@dataclass
class GGMLFunction:
    arg_types: List[Type]
    func: Callable

def get_rows(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_get_rows(ctx0, a.ptr, b.ptr))

def mul_mat(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    if not (a.shape[0] == b.shape[0] and b.shape[2] % a.shape[2] == 0 and b.shape[3] % a.shape[3] == 0):
        logging.warning(f"Incompatible shapes for matrix multiplication: {a.shape} and {b.shape}")
    
    return Tensor.from_tensor_ptr(result_name, gen.ggml_mul_mat(ctx0, a.ptr, b.ptr))

def rms_norm(ctx0: ggml_context_p, result_name: str, a: Tensor, eps: float):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_rms_norm(ctx0, a.ptr, eps))

def reshape_2d(ctx0: ggml_context_p, result_name: str, a: Tensor, ne0: int, ne1: int):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_reshape_2d(ctx0, a.ptr, ne0, ne1))

def reshape_3d(ctx0: ggml_context_p, result_name: str, a: Tensor, x: int, y: int, z: int):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_reshape_3d(ctx0, a.ptr, x, y, z))

def mul(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_mul(ctx0, a.ptr, b.ptr))

def add(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_add(ctx0, a.ptr, b.ptr))

def sub(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_sub(ctx0, a.ptr, b.ptr))

def div(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_div(ctx0, a.ptr, b.ptr))

def rope(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor, n_rot: int, mode: int, n_orig_ctx: int, freq_base: float, freq_scale: float, ext_factor: float, attn_factor: float, beta_fast: float, beta_slow: float):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_rope_ext(ctx0, a.ptr, b.ptr, ctypes.POINTER(ggml_tensor)(), n_rot, mode, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow))

def permute(ctx0: ggml_context_p, result_name: str, a: Tensor, axis0: int, axis1: int, axis2: int, axis3: int):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_permute(ctx0, a.ptr, axis0, axis1, axis2, axis3))

def soft_max(ctx0: ggml_context_p, result_name: str, a: Tensor, mask: Tensor, scale: float):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_soft_max_ext(ctx0, a.ptr, mask.ptr, scale, 0.0))

def cont(ctx0: ggml_context_p, result_name: str, a: Tensor):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_cont(ctx0, a.ptr))

def cont_2d(ctx0: ggml_context_p, result_name: str, a: Tensor, x: int, y: int):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_cont_2d(ctx0, a.ptr, x, y))

def transpose(ctx0: ggml_context_p, result_name: str, a: Tensor):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_transpose(ctx0, a.ptr))

def silu(ctx0: ggml_context_p, result_name: str, a: Tensor):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_silu(ctx0, a.ptr))

def view_1d(ctx0: ggml_context_p, result_name: str, a: Tensor, ne: int, offset: int):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_view_1d(ctx0, a.ptr, ne, offset))

def view_2d(ctx0: ggml_context_p, result_name: str, a: Tensor, ne0: int, ne1: int, nb1: int, offset: int):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_view_2d(ctx0, a.ptr, ne0, ne1, nb1, offset))

def view_3d(ctx0: ggml_context_p, result_name: str, a: Tensor, ne0: int, ne1: int, ne2: int, nb1: int, nb2: int, offset: int):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_view_3d(ctx0, a.ptr, ne0, ne1, ne2, nb1, nb2, offset))

def cpy(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, gen.ggml_cpy(ctx0, a.ptr, b.ptr))

GGML_FUNCTIONS = {
    "get_rows": GGMLFunction([Tensor, Tensor], get_rows),
    "mul_mat": GGMLFunction([Tensor, Tensor], mul_mat),
    "rms_norm": GGMLFunction([Tensor, float], rms_norm),
    "reshape_2d": GGMLFunction([Tensor, int, int], reshape_2d),
    "reshape_3d": GGMLFunction([Tensor, int, int, int], reshape_3d),
    "mul": GGMLFunction([Tensor, Tensor], mul),
    "add": GGMLFunction([Tensor, Tensor], add),
    "sub": GGMLFunction([Tensor, Tensor], sub),
    "div": GGMLFunction([Tensor, Tensor], div),
    "rope": GGMLFunction([Tensor, Tensor, int, int, int, float, float, float, float, float, float], rope),
    "permute": GGMLFunction([Tensor, int, int, int, int], permute),
    "soft_max": GGMLFunction([Tensor, Tensor, float], soft_max),
    "cont": GGMLFunction([Tensor], cont),
    "cont_2d": GGMLFunction([Tensor, int, int], cont_2d),
    "transpose": GGMLFunction([Tensor], transpose),
    "silu": GGMLFunction([Tensor], silu),
    "view_1d": GGMLFunction([Tensor, int, int], view_1d),
    "view_2d": GGMLFunction([Tensor, int, int, int, int], view_2d),
    "view_3d": GGMLFunction([Tensor, int, int, int, int, int, int], view_3d),
    "cpy": GGMLFunction([Tensor, Tensor], cpy),
}
