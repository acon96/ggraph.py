from __future__ import annotations
from typing import List, Type, Callable, Optional, Any
from dataclasses import dataclass
import logging
import ctypes
import functools

import numpy as np
import numpy.typing as npt
from gguf.gguf_reader import ReaderTensor, ReaderField

from gglm.wrapper.gen import *

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
}
