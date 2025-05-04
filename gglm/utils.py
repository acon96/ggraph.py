
from __future__ import annotations
from typing import Optional, Dict, List, Union, TypeAlias, Any, TypeVar
from dataclasses import dataclass, field
import functools
import ctypes

from lark import Token
import numpy as np
import numpy.typing as npt
import ggml
from ggml.utils import GGML_TYPE
from gguf.gguf_reader import ReaderTensor, ReaderField
from gguf.quants import quant_shape_from_byte_shape

from exo.inference.shard import Shard

class Tensor:
    name: str
    type: int
    shape: List[int]
    is_input: bool
    is_view: bool
    ptr: Optional[ggml.ggml_tensor_p]

    def __init__(self, *, 
                 name: str, type: int, shape: List[int],
                 is_input: bool = False, is_view: bool = False, 
                 ptr: Optional[ggml.ggml_tensor_p] = None):
        self.name = name
        self.type = type
        self.shape = ([int(x) for x in shape] + [1, 1, 1, 1])[:4]
        self.is_input = is_input
        self.is_view = is_view
        self.ptr = ptr

    @property
    def n_bytes(self) -> int:
        size = ggml.ggml_type_size(self.type)
        for dim in self.shape:
            size *= dim
        return size
    
    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name}, shape={self.shape}, type={self.type})"
    
    def __repr__(self):
        return str(self)
    
    @classmethod
    def from_reader_tensor(cls, reader_tensor: ReaderTensor) -> Tensor:
        return cls(
            name=reader_tensor.name,
            type=reader_tensor.tensor_type.value,
            shape=reader_tensor.shape,
            is_input=False,
        )
    
    @classmethod
    def from_tensor_ptr(cls, name: str, ptr: ggml.ggml_tensor_p) -> Tensor:
        return cls(
            name=name,
            type=ptr.contents.type,
            shape=list(ptr.contents.ne),
            is_input=False,
            ptr=ptr,
            is_view=(ptr.contents.view_src != ctypes.POINTER(ggml.ggml_tensor)())
        )

class ParseError(Exception):
    def __init__(self, message: str, token: Token | None):
        super().__init__(message)

        if isinstance(token, Token):
            self.line = token.line
            self.column = token.column
        else:
            self.line = "unknown"
            self.column = "unknown"
    def __str__(self) -> str:
        return f"{self.args[0]} at {self.line}:{self.column}"
    
    def override_pos(self, line: int, column: int):
        self.line = line
        self.column = column

        return self
        
@dataclass(kw_only=True)
class GGMLContextParams:
    """Stores parameters related to the current GGML execution context"""
    shard: Optional[Shard]
    n_threads: int
    n_batches: int
    n_ctx: int
    enable_flash_attn: bool
    yarn_ext_factor: float
    yarn_attn_factor: float
    yarn_beta_fast: float
    yarn_beta_slow: float
    rope_freq_base: float
    rope_freq_scale: float
    rope_scaling_type: int
    type_k: int
    type_v: int


@dataclass(kw_only=True)
class ParseContext:
    model: str
    source: str
    context_params: GGMLContextParams
    model_params: ModelParams
    created_tensors: List[Tensor] = field(default_factory=lambda: [])
    intermediate_tensors: List[Tensor] = field(default_factory=lambda: [])
    gguf_tensors: Dict[str, Tensor] = field(default_factory=lambda: {})
    graph: Dict[str, Tensor] = field(default_factory=lambda: {})
    ast: List[ASTNode] = field(default_factory=lambda: [])
    cur_layer: Optional[int] = None

class ASTNode:
    """
    Represents a node in the abstract syntax tree (AST) of a GGML model. 
    Contains a reference to its source token and the current parsing context to allow easy reporting of syntax errors.
    """
    ctx: ParseContext
    source_token: Token

    def __init__(self, source_token: Token, ctx: ParseContext):
        self.source_token = source_token
        self.ctx = ctx

    def resolve_param(self, name: str, *, raise_error: bool = True) -> Optional[int | float]:
        if name.startswith("params."):
            if name.startswith("params.context."):
                return getattr(self.ctx.context_params, name[len("params.context."):])
            elif name.startswith("params.model."):
                return getattr(self.ctx.model_params, name[len("params.model."):])
            
        try:
            return int(name)
        except ValueError:
            pass

        try:
            return float(name)
        except ValueError:
            pass
    
        if raise_error:
            raise ParseError(f"Unknown param {name}", self.source_token)
        
        return None
    
    def resolve_tensor(self, name: str, *, raise_error: bool = True) -> Optional[Tensor]:
        if "%d" in name:
            name = name % self.ctx.cur_layer
        
        if name in self.ctx.graph:
            return self.ctx.graph[name]
        
        if name in self.ctx.gguf_tensors:
            return self.ctx.gguf_tensors[name]

        if raise_error:
            raise ParseError(f"Unknown tensor {name}", self.source_token)
        
        return None

GraphArg: TypeAlias = Union[Tensor, int, str]

class ModelParams:
    """Provides an interface for consistently accessing model parameters from a GGUF file"""
    def __init__(self, params: Dict[str, ReaderField]):
        self._data = params
        self._prefix = str(params["general.architecture"].contents()) + "."

    def __getitem__(self, key: str) -> str:
        return str(self._data[self._prefix + key].contents())
    
    T = TypeVar('T')
    def get(self, key: str, default: T = None) -> str | T:
        try:
            return self[key]
        except KeyError:
            return default

    @property
    def n_ctx(self) -> int:
        return int(self["context_length"])
    
    @property
    def n_embd(self) -> int:
        return int(self["embedding_length"])
    
    @property
    def n_layer(self) -> int:
        return int(self["block_count"])
    
    @property
    def n_expert(self) -> int:
        return int(self["expert_count"])
    
    @property
    def n_expert_used(self) -> int:
        return int(self["expert_used_count"])
    
    @property
    def n_ffn(self) -> int:
        return int(self["feed_forward_length"])
    
    @property
    def n_head(self) -> int:
        return int(self["attention.head_count"])
    
    @property
    def n_head_kv(self) -> int:
        return int(self.get("attention.head_count_kv", self.n_head))
    
    @property
    def n_embd_k_gqa(self) -> int:
        return self.n_embd_head_k * self.n_head_kv
    
    @property
    def n_embd_v_gqa(self) -> int:
        return self.n_embd_head_v * self.n_head_kv
    
    @property
    def rope_finetuned(self) -> bool:
        return self.get("rope.scaling.finetuned") == "true"
    
    @property
    def rope_freq_base(self) -> float:
        return float(self.get("rope.freq_base", 10000.0))
    
    @property
    def rope_freq_scale(self) -> float:
        return float(self.get("rope.scaling.factor", 1.0))
    
    @property
    def rope_scaling_type(self) -> str:
        return self.get("rope.scaling.type", "linear")
    
    @property
    def rope_attn_factor(self) -> float:
        return float(self.get("rope.scaling.attn_factor", 1.0))
    
    @property
    def n_embd_head_k(self) -> int:
        return int(self.get("attention.key_length", self.n_embd // self.n_head))
    
    @property
    def n_embd_head_v(self) -> int:
        return int(self.get("attention.value_length", self.n_embd // self.n_head))
    
    @property
    def n_rot(self) -> int:
        return int(self.get("rope.dimension_count", self.n_embd_head_k))
    
    @property
    def n_vocab(self) -> int:
        return int(self.get("vocab_size", len(self._data["tokenizer.ggml.tokens"].parts)))
    
    @property
    def f_norm_rms_eps(self) -> float:
        return float(self["attention.layer_norm_rms_epsilon"])

    
    def to_default_ggml_context_params_dict(self) -> Dict[str, Any]:
        """Returns the default values for the context, provided in the gguf file"""
        return dict(
            n_ctx=self.n_ctx,
            rope_freq_base=self.rope_freq_base,
            rope_freq_scale=self.rope_freq_scale,
            rope_scaling_type=self.get("rope.scaling.type", 0),
        )
    
def ggml_tensor_size(shape: List[int], ggml_type: Optional[int] = None):
    tensor_overhead = ggml.ggml_tensor_overhead()

    if ggml_type:
        element_size = ggml.ggml_type_size(ggml_type)
    else:
        element_size = 1

    num_elements = functools.reduce(lambda x, y: x * y, shape, 1)
    return (num_elements * element_size) + tensor_overhead

def ensure_args(function_name: str, args: List[Tensor | int | float | str | None], types: List[type], token: Token) -> None:
    if len(args) != len(types):
        raise ParseError(f"Error in function call '{function_name}'. Expected {len(types)} arguments but got {len(args)}", token)
    
    for idx, (arg, expected_type) in enumerate(zip(args, types)):
        if not isinstance(arg, expected_type):
            raise ParseError(f"Error in function call '{function_name}'. Expected parameter {idx + 1} to be of type {expected_type.__name__}, but got {type(arg).__name__}", token)

GGML_TYPE_TO_CTYPE = {
    ggml.GGML_TYPE_F32: ctypes.c_uint32,
    ggml.GGML_TYPE_F16: ctypes.c_uint16,
    ggml.GGML_TYPE_I8: ctypes.c_int8,
    ggml.GGML_TYPE_I16: ctypes.c_int16,
    ggml.GGML_TYPE_I32: ctypes.c_int32
}

def create_tensor_from_gguf_shape(shape: List[int], ctx: ggml.ggml_context_p, tensor: Tensor) -> None:
    """Create a new ggml tensor with data copied from a numpy array. The provided tensor is updated to contain the pointer to the GGML tensor"""
    if tensor.type in GGML_TYPE_TO_CTYPE:
        shape = list(reversed(shape))
    else:
        shape = list(reversed(quant_shape_from_byte_shape(shape, tensor.type)))
        tensor.shape = (shape + [1, 1, 1, 1])[:4]
        
    tensor_ptr = ggml.ggml_new_tensor(
        ctx,
        tensor.type,
        len(shape),
        (ctypes.c_int64 * len(shape))(*shape),
    )

    ggml.ggml_set_name(tensor_ptr, tensor.name.encode())
    tensor.ptr = tensor_ptr


def set_tensor_from_numpy(x: npt.NDArray[Any], tensor: Tensor) -> None:
    """Create a new ggml tensor with data copied from a numpy array. The provided tensor is updated to contain the pointer to the GGML tensor"""
    if ggml.ggml_get_data(tensor.ptr):
        n_elements = ggml.ggml_nelements(tensor.ptr)
    
        if tensor.type in GGML_TYPE_TO_CTYPE:
            ctypes_type = GGML_TYPE_TO_CTYPE[tensor.type]
        else:
            ctypes_type = ctypes.c_byte
            n_elements = functools.reduce(lambda x, y: x * y, x.shape)

        n_bytes = n_elements * ctypes.sizeof(ctypes_type)

        ggml.ggml_backend_tensor_set(tensor.ptr, x.ctypes.data_as(ctypes.c_void_p), 0, n_bytes)
    else:
        raise ValueError("Tensor data is None")
    

def get_tensor_to_numpy(tensor: Tensor) -> npt.NDArray[Any]:
    n_elements = ggml.ggml_nelements(tensor.ptr)
    n_bytes = ggml.ggml_nbytes(tensor.ptr)
    result_buffer = (GGML_TYPE_TO_CTYPE[GGML_TYPE(tensor.type)] * n_elements)()

    ggml.ggml_backend_tensor_get(tensor.ptr, result_buffer, 0, n_bytes)

    return np.ctypeslib.as_array(result_buffer, list(reversed(tensor.shape)))