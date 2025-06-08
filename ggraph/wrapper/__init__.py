from __future__ import annotations
from typing import List, Type, Callable, Optional, Any
from dataclasses import dataclass
import logging
import ctypes
import functools

import numpy as np
import numpy.typing as npt
from gguf.gguf_reader import ReaderTensor, ReaderField

from ggraph.wrapper.gen import *

GGML_TYPE_TO_CTYPE = {
    GGML_TYPE_F32: ctypes.c_float,
    GGML_TYPE_I8: ctypes.c_int8,
    GGML_TYPE_I16: ctypes.c_int16,
    GGML_TYPE_I32: ctypes.c_int32
}

def set_tensor_from_numpy(x: npt.NDArray[Any], tensor: Tensor) -> None:
    """Copy data from a numpy array to a tensor"""
    if ggml_get_data(tensor.ptr):
        n_bytes = ggml_nbytes(tensor.ptr)
        src_ptr = x.ctypes.data_as(ctypes.c_void_p)
        ggml_backend_tensor_set(tensor.ptr, src_ptr, 0, n_bytes)
    else:
        raise ValueError("Tensor data is None")
    

def get_tensor_to_numpy(tensor: Tensor) -> npt.NDArray[Any]:
    """Retrieve data from a tensor and convert it to a numpy array"""
    n_bytes = ggml_nbytes(tensor.ptr)
    n_dims = ggml_n_dims(tensor.ptr)
    shape = tensor.shape[:n_dims]

    n_elems = 1
    for n in shape:
        n_elems *= n

    result_buffer_type = GGML_TYPE_TO_CTYPE.get(tensor.type)
    if result_buffer_type:
        result_buffer = (result_buffer_type * n_elems)()
        ggml_backend_tensor_get(tensor.ptr, result_buffer, 0, n_bytes)
    else:
        quantized_buffer = (ctypes.c_ubyte * n_bytes)()
        result_buffer = (ctypes.c_float * n_elems)()
        ggml_backend_tensor_get(tensor.ptr, ctypes.cast(quantized_buffer, ctypes.c_void_p), 0, n_bytes)

        if tensor.type == GGML_TYPE_F16:
            ggml_fp16_to_fp32_row(
                ctypes.cast(quantized_buffer, ctypes.POINTER(ggml_fp16_t)),
                ctypes.cast(result_buffer, ctypes.POINTER(ctypes.c_float)),
                n_elems
            )
        elif ggml_is_quantized(tensor.type):
            quantized_type: ggml_type_traits = ggml_get_type_traits(tensor.type)
            quantized_type.to_float(
                ctypes.cast(quantized_buffer, ctypes.c_void_p),
                ctypes.cast(result_buffer, ctypes.POINTER(ctypes.c_float)),
                n_elems
            )
        else:
            raise ValueError(f"Unsupported tensor type: {tensor.type}")

    return np.ctypeslib.as_array(result_buffer).reshape(shape)

def ggml_tensor_size(shape: List[int], ggml_type: Optional[int] = None):
    tensor_overhead = ggml_tensor_overhead()

    if ggml_type:
        element_size = ggml_type_size(ggml_type)
    else:
        element_size = 1

    num_elements = functools.reduce(lambda x, y: x * y, shape, 1)
    return (num_elements * element_size) + tensor_overhead

class Tensor:
    name: str
    type: int
    n_dims: int
    shape: List[int]
    is_input: bool
    is_view: bool
    is_loaded: bool
    is_cache: bool
    _ptr: Optional[ggml_tensor_p]
    _ctx: Optional[ggml_context_p]

    def __init__(self, *, 
                 name: str, type: int, shape: List[int],
                 is_input: bool = False, is_view: bool = False,
                 is_loaded: bool = False, is_cache: bool = False,
                 ptr: Optional[ggml_tensor_p] = None,
                 ctx: Optional[ggml_context_p] = None):
        self.name = name
        self.type = type
        self.n_dims = len(shape)
        self.shape = ([int(x) for x in shape] + [1, 1, 1, 1])[:4]
        self.is_input = is_input
        self.is_view = is_view
        self.is_loaded = is_loaded
        self.is_cache = is_cache
        self._ptr = ptr
        self._ctx = ctx
    
    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name}, shape={self.shape}, type={self.type})"
    
    def __repr__(self):
        return str(self)
    
    def allocate(self):
        if self._ptr is None:
            if self._ctx is None:
                raise ValueError("Context is not set for tensor allocation")
            
            n_dims = len(self.shape)
            tensor_ptr = ggml_new_tensor(self._ctx, self.type, n_dims, (ctypes.c_int64 * n_dims)(*self.shape))
            ggml_set_name(tensor_ptr, self.name.encode())

            if self.is_input:
                ggml_set_input(tensor_ptr)
            
            self._ptr = tensor_ptr
    
    @property
    def n_bytes(self) -> int:
        size = ggml_type_size(self.type)
        for dim in self.shape:
            size *= dim
        return size
    
    @property
    def n_bytes_ctx(self) -> int:
        return ggml_tensor_size(self.shape, self.type)
    
    @property
    def ptr(self) -> ggml_tensor_p:
        if self._ptr is None:
            raise ValueError("Attempt to access ptr of a tensor that is not loaded")
        return self._ptr
    
    @ptr.setter
    def ptr(self, new_value: ggml_tensor_p):
        self._ptr = new_value
    
    @property
    def data(self):
        return get_tensor_to_numpy(self)
    
    @data.setter
    def data(self, new_value):
        set_tensor_from_numpy(new_value, self)
    
    @classmethod
    def from_reader_tensor(cls, reader_tensor: ReaderTensor) -> Tensor:
        return cls(
            name=reader_tensor.name,
            type=reader_tensor.tensor_type.value,
            shape=reader_tensor.shape,
            is_input=False,
            is_loaded=True,
        )
    
    @classmethod
    def from_tensor_ptr(cls, name: str, ptr: ggml_tensor_p | None) -> Tensor:
        if ptr is None:
            raise ValueError("Attempt to create tensor from null pointer")
        return cls(
            name=name,
            type=ptr.contents.type,
            shape=list(ptr.contents.ne),
            is_input=False,
            ptr=ptr,
            is_view=(ptr.contents.view_src != ctypes.POINTER(ggml_tensor)())
        )
    
class VariableShapeTensor(Tensor):
    lowering_ctx: object
    _shape: List[object]

    def __init__(self, *,
                 name: str, type: int, shape: List[object],
                 is_input: bool = False, is_view: bool = False,
                 is_loaded: bool = False, is_cache: bool = False,
                 ptr: Optional[ggml_tensor_p] = None,
                 ctx: Optional[ggml_context_p] = None):
        self.name = name
        self.type = type
        self.n_dims = 4
        self._shape = shape
        self.is_input = is_input
        self.is_view = is_view
        self.is_loaded = is_loaded
        self.is_cache = is_cache
        self._ptr = ptr
        self._ctx = ctx

    @property
    def shape(self) -> List[int]:
        return [e.resolve_as_int(lowering_ctx=self.lowering_ctx) for e in self._shape]
    
    @shape.setter
    def shape(self, new_shape: List[int]):
        raise ValueError("Cannot set shape of VariableShapeTensor")

class GGMLContext:
    ctx_size: int
    ctx: ggml_context_p
    backend: ggml_backend_p
    # backend_buffer: ggml_backend_buffer_p
    tensors: List[Tensor]

    def __init__(self, *, ctx_size: Optional[int] = None, max_tensors: Optional[int] = None, graph_size: Optional[int] = None):
        if ctx_size is None:
            ctx_size = 0
            if max_tensors is not None:
                ctx_size += max_tensors * ggml_tensor_overhead()
            if graph_size:
                ctx_size += ggml_graph_overhead_custom(graph_size, False)

        self.ctx_size = ctx_size
        init_params = ggml_init_params(mem_size=ctx_size, mem_buffer=None, no_alloc=True)
        self.ctx = ggml_init(init_params)
        self.tensors = []
        
        if self.ctx is None:
            raise RuntimeError("Failed to initialize GGML context")

    def add_reader_tensor(self, reader_tensor: ReaderTensor) -> Tensor:
        tensor = Tensor.from_reader_tensor(reader_tensor)
        tensor._ctx = self.ctx
        self.tensors.append(tensor)
        return tensor

    def add_tensor_from_ptr(self, name: str, ptr: ggml_tensor_p) -> Tensor:
        tensor = Tensor.from_tensor_ptr(name, ptr)
        tensor._ctx = self.ctx
        self.tensors.append(tensor)
        return tensor
    
    def add_tensor_with_variable_shape(self, name: str, shape: List[object], type: int, **kwargs) -> VariableShapeTensor:
        tensor = VariableShapeTensor(name=name, type=type, shape=shape, **kwargs)
        tensor._ctx = self.ctx
        self.tensors.append(tensor)
        return tensor
        
    def reset(self):
        ggml_reset(self.ctx)

        for tensor in self.tensors:
            tensor._ptr = None

    def reallocate(self, lowering_ctx: Optional[object] = None):
        self.reset()

        for tensor in self.tensors:
            if isinstance(tensor, VariableShapeTensor) and lowering_ctx:
                tensor.lowering_ctx = lowering_ctx
            tensor.allocate()
        
    def __del__(self):
        """garbage collect the context"""
        if self.ctx is not None:
            ggml_free(self.ctx)



@dataclass
class GGMLFunction:
    arg_types: List[Type]
    func: Callable

def get_rows(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, ggml_get_rows(ctx0, a.ptr, b.ptr))

def mul_mat(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    if not (a.shape[0] == b.shape[0] and b.shape[2] % a.shape[2] == 0 and b.shape[3] % a.shape[3] == 0):
        logging.warning(f"Incompatible shapes for matrix multiplication: {a.shape} and {b.shape}")
    
    return Tensor.from_tensor_ptr(result_name, ggml_mul_mat(ctx0, a.ptr, b.ptr))

def rms_norm(ctx0: ggml_context_p, result_name: str, a: Tensor, eps: float):
    return Tensor.from_tensor_ptr(result_name, ggml_rms_norm(ctx0, a.ptr, eps))

def reshape_2d(ctx0: ggml_context_p, result_name: str, a: Tensor, ne0: int, ne1: int):
    return Tensor.from_tensor_ptr(result_name, ggml_reshape_2d(ctx0, a.ptr, ne0, ne1))

def reshape_3d(ctx0: ggml_context_p, result_name: str, a: Tensor, x: int, y: int, z: int):
    return Tensor.from_tensor_ptr(result_name, ggml_reshape_3d(ctx0, a.ptr, x, y, z))

def mul(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, ggml_mul(ctx0, a.ptr, b.ptr))

def add(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, ggml_add(ctx0, a.ptr, b.ptr))

def sub(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, ggml_sub(ctx0, a.ptr, b.ptr))

def div(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, ggml_div(ctx0, a.ptr, b.ptr))

def rope(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor, n_rot: int, mode: int, n_orig_ctx: int, freq_base: float, freq_scale: float, ext_factor: float, attn_factor: float, beta_fast: float, beta_slow: float):
    return Tensor.from_tensor_ptr(result_name, ggml_rope_ext(ctx0, a.ptr, b.ptr, ctypes.POINTER(ggml_tensor)(), n_rot, mode, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow))

def permute(ctx0: ggml_context_p, result_name: str, a: Tensor, axis0: int, axis1: int, axis2: int, axis3: int):
    return Tensor.from_tensor_ptr(result_name, ggml_permute(ctx0, a.ptr, axis0, axis1, axis2, axis3))

def soft_max(ctx0: ggml_context_p, result_name: str, a: Tensor, mask: Tensor, scale: float):
    return Tensor.from_tensor_ptr(result_name, ggml_soft_max_ext(ctx0, a.ptr, mask.ptr, scale, 0.0))

def cont(ctx0: ggml_context_p, result_name: str, a: Tensor):
    return Tensor.from_tensor_ptr(result_name, ggml_cont(ctx0, a.ptr))

def cont_2d(ctx0: ggml_context_p, result_name: str, a: Tensor, ne0: int, ne1: int):
    return Tensor.from_tensor_ptr(result_name, ggml_cont_2d(ctx0, a.ptr, ne0, ne1))

def transpose(ctx0: ggml_context_p, result_name: str, a: Tensor):
    return Tensor.from_tensor_ptr(result_name, ggml_transpose(ctx0, a.ptr))

def silu(ctx0: ggml_context_p, result_name: str, a: Tensor):
    return Tensor.from_tensor_ptr(result_name, ggml_silu(ctx0, a.ptr))

def view_1d(ctx0: ggml_context_p, result_name: str, a: Tensor, ne: int, offset: int):
    return Tensor.from_tensor_ptr(result_name, ggml_view_1d(ctx0, a.ptr, ne, offset))

def view_2d(ctx0: ggml_context_p, result_name: str, a: Tensor, ne0: int, ne1: int, nb1: int, offset: int):
    # logging.debug(f"{a.name} [{a.shape}] ({list(a.ptr.contents.nb)}) view_2d: ne0={ne0}, ne1={ne1}, nb1={nb1}, offset={offset} total_bytes={ggml_nbytes(a.ptr)}")
    return Tensor.from_tensor_ptr(result_name, ggml_view_2d(ctx0, a.ptr, ne0, ne1, nb1, offset))

def view_3d(ctx0: ggml_context_p, result_name: str, a: Tensor, ne0: int, ne1: int, ne2: int, nb1: int, nb2: int, offset: int):
    return Tensor.from_tensor_ptr(result_name, ggml_view_3d(ctx0, a.ptr, ne0, ne1, ne2, nb1, nb2, offset))

def cpy(ctx0: ggml_context_p, result_name: str, a: Tensor, b: Tensor):
    return Tensor.from_tensor_ptr(result_name, ggml_cpy(ctx0, a.ptr, b.ptr))

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

    # Additional ops (AI generated so might not be correct)
    "dup": GGMLFunction([Tensor], lambda ctx0, result_name, a: Tensor.from_tensor_ptr(result_name, ggml_dup(ctx0, a.ptr))),
    "add1": GGMLFunction([Tensor, Tensor], lambda ctx0, result_name, a, b: Tensor.from_tensor_ptr(result_name, ggml_add1(ctx0, a.ptr, b.ptr))),
    # "acc": GGMLFunction([Tensor, Tensor], lambda ctx0, result_name, a, b: Tensor.from_tensor_ptr(result_name, ggml_acc(ctx0, a.ptr, b.ptr))),
    "sqr": GGMLFunction([Tensor], lambda ctx0, result_name, a: Tensor.from_tensor_ptr(result_name, ggml_sqr(ctx0, a.ptr))),
    "log": GGMLFunction([Tensor], lambda ctx0, result_name, a: Tensor.from_tensor_ptr(result_name, ggml_log(ctx0, a.ptr))),
    "sin": GGMLFunction([Tensor], lambda ctx0, result_name, a: Tensor.from_tensor_ptr(result_name, ggml_sin(ctx0, a.ptr))),
    "cos": GGMLFunction([Tensor], lambda ctx0, result_name, a: Tensor.from_tensor_ptr(result_name, ggml_cos(ctx0, a.ptr))),
    "sum": GGMLFunction([Tensor], lambda ctx0, result_name, a: Tensor.from_tensor_ptr(result_name, ggml_sum(ctx0, a.ptr))),
    "sum_rows": GGMLFunction([Tensor], lambda ctx0, result_name, a: Tensor.from_tensor_ptr(result_name, ggml_sum_rows(ctx0, a.ptr))),
    "mean": GGMLFunction([Tensor], lambda ctx0, result_name, a: Tensor.from_tensor_ptr(result_name, ggml_mean(ctx0, a.ptr))),
    "argmax": GGMLFunction([Tensor], lambda ctx0, result_name, a: Tensor.from_tensor_ptr(result_name, ggml_argmax(ctx0, a.ptr))),
    "count_equal": GGMLFunction([Tensor, Tensor], lambda ctx0, result_name, a, b: Tensor.from_tensor_ptr(result_name, ggml_count_equal(ctx0, a.ptr, b.ptr))),
    "repeat": GGMLFunction([Tensor, Tensor], lambda ctx0, result_name, a, b: Tensor.from_tensor_ptr(result_name, ggml_repeat(ctx0, a.ptr, b.ptr))),
    # "repeat_back": GGMLFunction([Tensor, Tensor], lambda ctx0, result_name, a, b: Tensor.from_tensor_ptr(result_name, ggml_repeat_back(ctx0, a.ptr, b.ptr))),
    "concat": GGMLFunction([Tensor, Tensor, int], lambda ctx0, result_name, a, b, dim: Tensor.from_tensor_ptr(result_name, ggml_concat(ctx0, a.ptr, b.ptr, dim))),
    # "silu_back": GGMLFunction([Tensor, Tensor], lambda ctx0, result_name, a, b: Tensor.from_tensor_ptr(result_name, ggml_silu_back(ctx0, a.ptr, b.ptr))),
    "norm": GGMLFunction([Tensor, float], lambda ctx0, result_name, a, eps: Tensor.from_tensor_ptr(result_name, ggml_norm(ctx0, a.ptr, eps))),
    # "rms_norm_back": GGMLFunction([Tensor, Tensor, float], lambda ctx0, result_name, a, b, eps: Tensor.from_tensor_ptr(result_name, ggml_rms_norm_back(ctx0, a.ptr, b.ptr, eps))),
    "group_norm": GGMLFunction([Tensor, int, float], lambda ctx0, result_name, a, n, eps: Tensor.from_tensor_ptr(result_name, ggml_group_norm(ctx0, a.ptr, n, eps))),
    "l2_norm": GGMLFunction([Tensor, float], lambda ctx0, result_name, a, eps: Tensor.from_tensor_ptr(result_name, ggml_l2_norm(ctx0, a.ptr, eps))),
    "mul_mat_id": GGMLFunction([Tensor, Tensor, int], lambda ctx0, result_name, a, b, id: Tensor.from_tensor_ptr(result_name, ggml_mul_mat_id(ctx0, a.ptr, b.ptr, id))),
    "out_prod": GGMLFunction([Tensor, Tensor], lambda ctx0, result_name, a, b: Tensor.from_tensor_ptr(result_name, ggml_out_prod(ctx0, a.ptr, b.ptr))),
    "scale": GGMLFunction([Tensor, float], lambda ctx0, result_name, a, s: Tensor.from_tensor_ptr(result_name, ggml_scale(ctx0, a.ptr, s))),
    # "set": GGMLFunction([Tensor, float], lambda ctx0, result_name, a, v: Tensor.from_tensor_ptr(result_name, ggml_set(ctx0, a.ptr, v))),
    "diag": GGMLFunction([Tensor], lambda ctx0, result_name, a: Tensor.from_tensor_ptr(result_name, ggml_diag(ctx0, a.ptr))),
    "diag_mask_inf": GGMLFunction([Tensor, int], lambda ctx0, result_name, a, n: Tensor.from_tensor_ptr(result_name, ggml_diag_mask_inf(ctx0, a.ptr, n))),
    "diag_mask_zero": GGMLFunction([Tensor, int], lambda ctx0, result_name, a, n: Tensor.from_tensor_ptr(result_name, ggml_diag_mask_zero(ctx0, a.ptr, n))),
    # "soft_max_back": GGMLFunction([Tensor, Tensor], lambda ctx0, result_name, a, b: Tensor.from_tensor_ptr(result_name, ggml_soft_max_back(ctx0, a.ptr, b.ptr))),
    # "rope_back": GGMLFunction([Tensor, Tensor, int, int, int, float, float, float, float, float, float], lambda ctx0, result_name, a, b, n_rot, mode, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow: Tensor.from_tensor_ptr(result_name, ggml_rope_back_ext(ctx0, a.ptr, b.ptr, ctypes.POINTER(ggml_tensor)(), n_rot, mode, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow))),
    "clamp": GGMLFunction([Tensor, float, float], lambda ctx0, result_name, a, min_val, max_val: Tensor.from_tensor_ptr(result_name, ggml_clamp(ctx0, a.ptr, min_val, max_val))),
    "conv_transpose_1d": GGMLFunction([Tensor, Tensor, int, int, int], lambda ctx0, result_name, a, b, s0, p0, d0: Tensor.from_tensor_ptr(result_name, ggml_conv_transpose_1d(ctx0, a.ptr, b.ptr, s0, p0, d0))),
    # "im2col": GGMLFunction([Tensor, int, int, int, int, int, int], lambda ctx0, result_name, a, s0, s1, k0, k1, p0, p1: Tensor.from_tensor_ptr(result_name, ggml_im2col(ctx0, a.ptr, s0, s1, k0, k1, p0, p1))),
    # "im2col_back": GGMLFunction([Tensor, int, int, int, int, int, int], lambda ctx0, result_name, a, s0, s1, k0, k1, p0, p1: Tensor.from_tensor_ptr(result_name, ggml_im2col_back(ctx0, a.ptr, s0, s1, k0, k1, p0, p1))),
    "conv_2d_dw": GGMLFunction([Tensor, Tensor, int, int, int, int, int, int], lambda ctx0, result_name, a, b, s0, s1, p0, p1, d0, d1: Tensor.from_tensor_ptr(result_name, ggml_conv_2d_dw(ctx0, a.ptr, b.ptr, s0, s1, p0, p1, d0, d1))),
    "pool_1d": GGMLFunction([Tensor, int, int, int, int], lambda ctx0, result_name, a, op, k0, s0, p0: Tensor.from_tensor_ptr(result_name, ggml_pool_1d(ctx0, a.ptr, op, k0, s0, p0))),
    "pool_2d": GGMLFunction([Tensor, int, int, int, int, int, int, int], lambda ctx0, result_name, a, op, k0, k1, s0, s1, p0, p1: Tensor.from_tensor_ptr(result_name, ggml_pool_2d(ctx0, a.ptr, op, k0, k1, s0, s1, p0, p1))),
    # "pool_2d_back": GGMLFunction([Tensor, Tensor, int, int, int, int, int, int, int, int], lambda ctx0, result_name, a, af, op, k0, k1, s0, s1, p0, p1: Tensor.from_tensor_ptr(result_name, ggml_pool_2d_back(ctx0, a.ptr, af.ptr, op, k0, k1, s0, s1, p0, p1))),
    "upscale": GGMLFunction([Tensor, int, int], lambda ctx0, result_name, a, scale_factor, mode: Tensor.from_tensor_ptr(result_name, ggml_upscale(ctx0, a.ptr, scale_factor, mode))),
    "pad": GGMLFunction([Tensor, int, float], lambda ctx0, result_name, a, p0, p1, p2, p3: Tensor.from_tensor_ptr(result_name, ggml_pad(ctx0, a.ptr, p0, p1, p2, p3))),
    "pad_reflect_1d": GGMLFunction([Tensor, int], lambda ctx0, result_name, a, p0, p1: Tensor.from_tensor_ptr(result_name, ggml_pad_reflect_1d(ctx0, a.ptr, p0, p1))),
    "arange": GGMLFunction([float, float, float], lambda ctx0, result_name, start, stop, step: Tensor.from_tensor_ptr(result_name, ggml_arange(ctx0, start, stop, step))),
    "timestep_embedding": GGMLFunction([Tensor, int, int, float, float], lambda ctx0, result_name, a, n, d, max_period, freq_base: Tensor.from_tensor_ptr(result_name, ggml_timestep_embedding(ctx0, a.ptr, n, d, max_period, freq_base))),
    "argsort": GGMLFunction([Tensor, int], lambda ctx0, result_name, a, order: Tensor.from_tensor_ptr(result_name, ggml_argsort(ctx0, a.ptr, order))),
    "leaky_relu": GGMLFunction([Tensor, float, bool], lambda ctx0, result_name, a, negative_slope, inplace: Tensor.from_tensor_ptr(result_name, ggml_leaky_relu(ctx0, a.ptr, negative_slope, inplace))),
    "flash_attn_ext": GGMLFunction([Tensor, Tensor, Tensor, Tensor, float, float, float], lambda ctx0, result_name, q, k, v, mask, scale, max_bias, logit_softcap: Tensor.from_tensor_ptr(result_name, ggml_flash_attn_ext(ctx0, q.ptr, k.ptr, v.ptr, mask.ptr, scale, max_bias, logit_softcap))),
    # "flash_attn_back": GGMLFunction([Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float], lambda ctx0, result_name, q, k, v, mask, d, scale, max_bias, logit_softcap: Tensor.from_tensor_ptr(result_name, ggml_flash_attn_back(ctx0, q.ptr, k.ptr, v.ptr, mask.ptr, d.ptr, scale, max_bias, logit_softcap))),
    # "ssm_conv": GGMLFunction([Tensor, Tensor, Tensor, Tensor, int], lambda ctx0, result_name, a, b, c, d, stride: Tensor.from_tensor_ptr(result_name, ggml_ssm_conv(ctx0, a.ptr, b.ptr, c.ptr, d.ptr, stride))),
    # "ssm_scan": GGMLFunction([Tensor, Tensor, Tensor, Tensor, int], lambda ctx0, result_name, a, b, c, d, stride: Tensor.from_tensor_ptr(result_name, ggml_ssm_scan(ctx0, a.ptr, b.ptr, c.ptr, d.ptr, stride))),
    # "win_part": GGMLFunction([Tensor, int, int], lambda ctx0, result_name, a, n, m: Tensor.from_tensor_ptr(result_name, ggml_win_part(ctx0, a.ptr, n, m))),
    # "win_unpart": GGMLFunction([Tensor, int, int], lambda ctx0, result_name, a, n, m: Tensor.from_tensor_ptr(result_name, ggml_win_unpart(ctx0, a.ptr, n, m))),
    "get_rel_pos": GGMLFunction([Tensor, int, int], lambda ctx0, result_name, a, qh, kh: Tensor.from_tensor_ptr(result_name, ggml_get_rel_pos(ctx0, a.ptr, qh, kh))),
    "add_rel_pos": GGMLFunction([Tensor, int, int], lambda ctx0, result_name, a, pw, ph: Tensor.from_tensor_ptr(result_name, ggml_add_rel_pos(ctx0, a.ptr, pw, ph))),
    "rwkv_wkv6": GGMLFunction([Tensor, Tensor, Tensor, Tensor, Tensor, int], lambda ctx0, result_name, a, b, c, d, e, f: Tensor.from_tensor_ptr(result_name, ggml_rwkv_wkv6(ctx0, a.ptr, b.ptr, c.ptr, d.ptr, e.ptr, f))),
    "gated_linear_attn": GGMLFunction([Tensor, Tensor, Tensor, Tensor, Tensor, float], lambda ctx0, result_name, k, q, v, g, state, scale: Tensor.from_tensor_ptr(result_name, ggml_gated_linear_attn(ctx0, k.ptr, q.ptr, v.ptr, g.ptr, state.ptr, scale))),
    "rwkv_wkv7": GGMLFunction([Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], lambda ctx0, result_name, r, w, k, v, a, b, state: Tensor.from_tensor_ptr(result_name, ggml_rwkv_wkv7(ctx0, r.ptr, w.ptr, k.ptr, v.ptr, a.ptr, b.ptr, state.ptr))),
    "unary": GGMLFunction([Tensor, int], lambda ctx0, result_name, a, op: Tensor.from_tensor_ptr(result_name, ggml_unary(ctx0, a.ptr, op))),
    # "map_custom1": GGMLFunction([Tensor, int, int, int, int], lambda ctx0, result_name, a, b, c, d, e: Tensor.from_tensor_ptr(result_name, ggml_map_custom1(ctx0, a.ptr, b, c, d, e))),
    # "map_custom2": GGMLFunction([Tensor, Tensor, int, int, int, int], lambda ctx0, result_name, a, b, c, d, e, f: Tensor.from_tensor_ptr(result_name, ggml_map_custom2(ctx0, a.ptr, b.ptr, c, d, e, f))),
    # "map_custom3": GGMLFunction([Tensor, Tensor, Tensor, int, int, int, int], lambda ctx0, result_name, a, b, c, d, e, f, g: Tensor.from_tensor_ptr(result_name, ggml_map_custom3(ctx0, a.ptr, b.ptr, c.ptr, d, e, f, g))),
    "cross_entropy_loss": GGMLFunction([Tensor, Tensor], lambda ctx0, result_name, a, b: Tensor.from_tensor_ptr(result_name, ggml_cross_entropy_loss(ctx0, a.ptr, b.ptr))),
    # "cross_entropy_loss_back": GGMLFunction([Tensor, Tensor, Tensor], lambda ctx0, result_name, a, b, c: Tensor.from_tensor_ptr(result_name, ggml_cross_entropy_loss_back(ctx0, a.ptr, b.ptr, c.ptr))),
    "opt_step_adamw": GGMLFunction([Tensor, Tensor, Tensor, Tensor, Tensor], lambda ctx0, result_name, a, b, c, d, e: Tensor.from_tensor_ptr(result_name, ggml_opt_step_adamw(ctx0, a.ptr, b.ptr, c.ptr, d.ptr, e.ptr))),
}
