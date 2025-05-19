from __future__ import annotations
from typing import Optional, Dict, Any, List
import enum
import os
import logging
import ctypes

import numpy as np
from gguf.gguf_reader import GGUFReader

from gglm.utils import Tensor, ModelParams, GGMLContextParams, ParseError
from gglm.models.parser import GGMLParser
from gglm.models.ast import produce_ggml_graph
from gglm.wrapper import ggml_context_p, ggml_cgraph_p, ggml_init_params, ggml_cgraph, ggml_tensor, gen

# from exo.inference.shard import Shard

class GGMLBackendType(enum.Enum):
    CPU = enum.auto()
    CUDA = enum.auto()
    VULKAN = enum.auto()

GGML_TYPE_TO_NUMPY_DTYPE = {
    gen.GGML_TYPE_F16: np.float16,
    gen.GGML_TYPE_F32: np.float32,
    gen.GGML_TYPE_I8: np.int8,
    gen.GGML_TYPE_I16: np.int16,
    gen.GGML_TYPE_I32: np.int32
}

class GGMLModel:
    """The base class for defining GGML model compute graphs and context buffers"""

    reader: GGUFReader
    context_params: GGMLContextParams
    model_params: ModelParams
    loaded_tensors: Dict[str, Tensor]
    created_tensors: Dict[str, Tensor]
    input_tensors: Dict[str, Tensor]
    ctx0: Optional[ggml_context_p]
    compute_graph: Optional[ggml_cgraph_p]

    def __init__(self, model_path: str, backend_type: GGMLBackendType = GGMLBackendType.CPU, **model_kwargs):
        # self.shard = shard
        self.reader = GGUFReader(model_path, mode="r")
        self.backend_type = backend_type

        gguf_kv = dict(**self.reader.fields)
        self.model_params = ModelParams(gguf_kv)
        params = self.model_params.to_default_ggml_context_params_dict()
        params.update(dict(
            # shard=shard,
            n_threads=4,
            n_batches=1,
            enable_flash_attn=False,
            yarn_ext_factor=-1.0,
            yarn_attn_factor=1.0,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            type_k=gen.GGML_TYPE_F16,
            type_v=gen.GGML_TYPE_F16,
        ))
        params.update(model_kwargs)
        self.context_params = GGMLContextParams(**params)

        arch = str(gguf_kv["general.architecture"].contents())
        try:
            self.parse_context = GGMLParser().parse(os.path.join(os.path.dirname(__file__), f"{arch}.ggml"), self.context_params, self.model_params)
        except ParseError as exception:
            raise RuntimeError("Failed to parse") from exception

        self.parse_context.gguf_tensors = {
            tensor.name: Tensor.from_reader_tensor(tensor) for tensor in self.reader.tensors
        }
        self.ctx0 = None
        self.backend = None
        self.inputs_buffer = None
        self.compute_graph = None
        self.loaded_tensors = {}
        self.created_tensors = {}
        self.input_tensors = {}

        ctx_size = self._get_ctx_size()
        logging.info(f"Init GGML with context size: {ctx_size / (1024 ** 2):,.2f} MB")
        self._init_ggml_ctx(ctx_size)

    def _get_ctx_size(self) -> int:
        """Calculate the total size of the GGML context buffer based on the parsed model file"""
        tensors_to_load = self.parse_context.gguf_tensors.values()
        tensors_to_create = self.parse_context.created_tensors
        intermediate_tensors = self.parse_context.intermediate_tensors

        model_bytes = 0
        for tensor in tensors_to_load:
            model_bytes += tensor.n_bytes_ctx

        context_bytes = 0
        for tensor in tensors_to_create:
            context_bytes += tensor.n_bytes_ctx

        for tensor in intermediate_tensors:
            if tensor.is_view:
                continue
            context_bytes += tensor.n_bytes_ctx

        num_tensors = len(tensors_to_load) + len(tensors_to_create) + len(intermediate_tensors)
        graph_bytes = gen.ggml_graph_overhead_custom(num_tensors * 5, False) + gen.ggml_tensor_overhead() * num_tensors
        
        return model_bytes + context_bytes + graph_bytes

    def _init_ggml_ctx(self, size: int):
        init_params = ggml_init_params(mem_size=size, mem_buffer=None, no_alloc=True)
        self.ctx0 = gen.ggml_init(init_params)

        if not self.ctx0:
            raise ValueError("Failed to initialize GGML!")
        
        match self.backend_type:
            case GGMLBackendType.CPU:
                self.backend = gen.ggml_backend_cpu_init()
            case GGMLBackendType.CUDA:
                try:
                    num_gpus = gen.ggml_backend_cuda_get_device_count()
                    if num_gpus == 0:
                        raise RuntimeError("Cannot use CUDA backend. No NVIDIA GPUs were detected!")
                    self.backend = gen.ggml_backend_cuda_init(0)
                except RuntimeError:
                    raise RuntimeError("This copy of ggml-py was not built with CUDA support!")
            case _:
                raise ValueError(f"Invalid Backend {self.backend_type}")

        if not self.backend:
            raise RuntimeError("Failed to initialize backend!")
        
    
    def _load_tensor_from_gguf(self, tensor: Tensor):
        logging.debug(f"Loading Tensor - name: {tensor.name}, shape: {tensor.shape}, type: {tensor.ptr.contents.type}")
        if not self.ctx0:
            raise ValueError("Context is not initialized yet! Cannot create tensor.")
        
        reader_tensors = [x for x in self.reader.tensors if x.name == tensor.name]
        if not reader_tensors:
            raise ValueError(f"Tensor {tensor.name} not found in GGUF file!")

        np_tensor = reader_tensors[0].data
        logging.debug(f"{np_tensor.dtype}, {np_tensor.shape=}")
        tensor.data = np_tensor

    def _create_empty_tensor(self, tensor: Tensor):
        if not self.ctx0:
            raise ValueError("Context is not initialized yet! Cannot create tensor.")
        
        n_dims = len(tensor.shape)
        tensor_name = tensor.name

        tensor_ptr = gen.ggml_new_tensor(self.ctx0, tensor.type, n_dims, (ctypes.c_int64 * n_dims)(*tensor.shape))
        gen.ggml_set_name(tensor_ptr, tensor_name.encode())

        tensor.ptr = tensor_ptr

        if tensor.is_loaded:
            self.loaded_tensors[tensor_name] = tensor
        else:
            self.created_tensors[tensor_name] = tensor

        if tensor.is_input:
            gen.ggml_set_input(tensor_ptr)
            self.input_tensors[tensor_name] = tensor

        return tensor

    def _setup_tensors(self) -> None:
        """Load the required tensors in memory. TODO: If a shard was provided, only load the desired layers"""
        if not self.backend:
            raise RuntimeError("Cannot set up tensors. Backend is not initialized")

        tensors_to_load: list[Tensor] = []
        for tensor in self.parse_context.gguf_tensors.values():
            if tensor.name not in self.loaded_tensors:
                self._create_empty_tensor(tensor)
                tensors_to_load.append(tensor)

        for tensor in self.parse_context.created_tensors:
            if tensor.name not in self.created_tensors:
                self._create_empty_tensor(tensor)

        self.inputs_buffer = gen.ggml_backend_alloc_ctx_tensors(self.ctx, self.backend)
        if not self.inputs_buffer:
            raise RuntimeError("Failed to allocate buffer to store inputs!")
        gen.ggml_backend_buffer_clear(self.inputs_buffer, 0)

        for tensor in tensors_to_load:
            self._load_tensor_from_gguf(tensor)

    def _build_forward(self):
        """Build the GGM Graph from the parsed AST and ensure the tensors are loaded"""
        logging.debug("Loading tensors...")
        self._setup_tensors()

        graph_size = (len(self.loaded_tensors) + len(self.created_tensors)) * 5
        logging.debug(f"Creating GGML graph with {graph_size} nodes...")
        gf = gen.ggml_new_graph_custom(self.ctx, graph_size, False)

        try:
            for node in self.parse_context.ast:
                produce_ggml_graph(self.ctx, gf, node)
        except ParseError as exception:
            raise RuntimeError("Failed to parse") from exception

        output_tensor = self.parse_context.graph.get("output")
        if not output_tensor:
            raise ValueError("Output tensor not found in the graph.")
        
        gen.ggml_set_name(output_tensor.ptr, b"result_output")
        
        logging.debug("Expanding graph...")
        gen.ggml_build_forward_expand(gf, output_tensor.ptr)
        gen.ggml_graph_dump_dot(gf, ctypes.POINTER(ggml_cgraph)(), b"graph.dot")

        self.compute_graph = gf
    
    def __call__(self, **kwargs: List[int | float]):
        """Run the GGML model with the provided input tensors"""
        if not self.compute_graph:
            self._build_forward()

        if not self.backend:
            raise RuntimeError("Cannot evaluate model without initializing the backend")

        for input_name in kwargs.keys():
            input_tensor = self.input_tensors[input_name]
            input_dtype = GGML_TYPE_TO_NUMPY_DTYPE[input_tensor.type]

            input_tensor.data = np.array(kwargs[input_name], dtype=input_dtype)

        backend_buffer_type = gen.ggml_backend_get_default_buffer_type(self.backend)
        if not backend_buffer_type:
            raise RuntimeError()
        
        allocr = gen.ggml_gallocr_new(backend_buffer_type)
        if not allocr:
            raise RuntimeError()
        
        gen.ggml_gallocr_alloc_graph(allocr, self.compute_graph)

        if self.backend_type == GGMLBackendType.CPU.value:
            gen.ggml_backend_cpu_set_n_threads(self.backend, 8)

        gen.ggml_backend_graph_compute(self.backend, self.compute_graph)

        output_ptr = gen.ggml_graph_get_tensor(self.compute_graph, b"result_output")
        assert output_ptr != ctypes.POINTER(ggml_tensor)()

        output = Tensor.from_tensor_ptr("result_output", output_ptr)
        logprobs = output.data.reshape(
            -1, output.data.shape[0]
        ).copy()

        return logprobs
    
    @property
    def ctx(self) -> ggml_context_p:
        if not self.ctx0:
            raise ValueError("Context is not initialized yet! Cannot create tensor.")
        return self.ctx0

    def __repr__(self):
        return f"{self.__class__.__name__}({self.loaded_tensors.keys()})"