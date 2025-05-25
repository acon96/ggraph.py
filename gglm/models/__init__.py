from __future__ import annotations
from typing import Optional, Dict, Any, List
import enum
import os
import logging
import ctypes

import numpy as np
from gguf.gguf_reader import GGUFReader

from gglm.utils import Tensor, ModelParams, ContextParams, BatchParams, ParseError
from gglm.models.parser import GGMLParser, ParseContext
from gglm.models.ast import LoweringContext, produce_ggml_graph
import gglm.wrapper as wrapper

# from exo.inference.shard import Shard

wrapper.ggml_backend_dev_type

class GGMLBackendType(enum.Enum):
    CPU = enum.auto()
    CUDA = enum.auto()
    VULKAN = enum.auto()

class GGMLContextType(enum.Enum):
    GGUF = enum.auto()
    COMPUTE = enum.auto()
    INPUT_OUTPUT = enum.auto()

GGML_TYPE_TO_NUMPY_DTYPE = {
    wrapper.GGML_TYPE_F16: np.float16,
    wrapper.GGML_TYPE_F32: np.float32,
    wrapper.GGML_TYPE_I8: np.int8,
    wrapper.GGML_TYPE_I16: np.int16,
    wrapper.GGML_TYPE_I32: np.int32
}

class GGMLModel:
    """The base class for defining GGML model compute graphs and context buffers"""

    reader: GGUFReader
    context_params: ContextParams
    model_params: ModelParams
    batch_params: BatchParams
    loaded_tensors: Dict[str, Tensor]
    created_tensors: Dict[str, Tensor]
    input_tensors: Dict[str, Tensor]
    ggml_ctx: dict[GGMLContextType, wrapper.GGMLContext]
    compute_graph: Optional[wrapper.ggml_cgraph_p]
    parse_context: ParseContext
    lowering_context: LoweringContext

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
            enable_flash_attn=False,
            yarn_ext_factor=-1.0,
            yarn_attn_factor=1.0,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            type_k=wrapper.GGML_TYPE_F16,
            type_v=wrapper.GGML_TYPE_F16,
        ))
        params.update(model_kwargs)
        self.context_params = ContextParams(**params)
        self.batch_params = BatchParams(n_tokens=self.context_params.n_ctx, kv_output_pos=0)

        arch = str(gguf_kv["general.architecture"].contents())
        try:
            self.parse_context = GGMLParser().parse(os.path.join(os.path.dirname(__file__), f"{arch}.ggml"), self.context_params, self.model_params)
        except ParseError as exception:
            raise RuntimeError("Failed to parse") from exception
        
        self.lowering_context = LoweringContext.from_parse_context(self.parse_context, self.batch_params)
        
        self._init_ggml_backend()
        
        self.ggml_ctx = {
            GGMLContextType.GGUF: self._load_model(),
            GGMLContextType.INPUT_OUTPUT: self._allocate_inputs(),
            GGMLContextType.COMPUTE: None,
        }

        self.compute_graph = None

    def _init_ggml_backend(self):
        match self.backend_type:
            case GGMLBackendType.CPU:
                self.backend = wrapper.ggml_backend_cpu_init()
            case GGMLBackendType.CUDA:
                try:
                    num_gpus = wrapper.ggml_backend_cuda_get_device_count()
                    if num_gpus == 0:
                        raise RuntimeError("Cannot use CUDA backend. No NVIDIA GPUs were detected!")
                    self.backend = wrapper.ggml_backend_cuda_init(0)
                except RuntimeError:
                    raise RuntimeError("This copy of ggml-py was not built with CUDA support!")
            case _:
                raise ValueError(f"Invalid Backend {self.backend_type}")

        if not self.backend:
            raise RuntimeError("Failed to initialize backend!")

    def _load_model(self) -> wrapper.GGMLContext:
        logging.info("Loading tensors...")
        # model_bytes = 0
        # for tensor in self.reader.tensors:
        #     model_bytes += tensor.n_bytes

        ctx_size = wrapper.ggml_tensor_overhead() * len(self.reader.tensors)
        gguf_context = wrapper.GGMLContext(ctx_size, backend=self.backend)
        tensors_and_data: list[tuple[Tensor, np.ndarray]] = []
        for t in self.reader.tensors:
            tensor = gguf_context.add_reader_tensor(t)
            tensor.allocate()
            tensors_and_data.append((tensor, t.data))

        self.model_buffer = wrapper.ggml_backend_alloc_ctx_tensors(gguf_context.ctx, self.backend)

        for tensor, np_tensor in tensors_and_data:
            logging.debug(f"Loading Tensor - name: {tensor.name}, shape: {tensor.shape}, type: {tensor.ptr.contents.type}")
            tensor.data = np_tensor
        
        return gguf_context

    def _get_ctx_size(self, ctx_type: GGMLContextType) -> int:
        """Calculate the total size of the GGML context buffer based on the parsed model file"""
        tensors_to_load = self.lowering_context.gguf_tensors.values()
        tensors_to_create = self.parse_context.created_tensors
        intermediate_tensors = self.lowering_context.intermediate_tensors

        if ctx_type == GGMLContextType.GGUF:
            model_bytes = 0
            for tensor in tensors_to_load:
                model_bytes += tensor.n_bytes
            return model_bytes
        elif ctx_type == GGMLContextType.INPUT_OUTPUT:
            context_bytes = 0
            for tensor in tensors_to_create:
                context_bytes += wrapper.ggml_tensor_size(tensor.shape, tensor.type)

            for tensor in intermediate_tensors:
                if tensor.is_view:
                    continue
                context_bytes += tensor.n_bytes_ctx
            return context_bytes
        elif ctx_type == GGMLContextType.COMPUTE:
            num_tensors = len(tensors_to_load) + len(tensors_to_create) + len(intermediate_tensors)
            graph_bytes = wrapper.ggml_graph_overhead_custom(num_tensors * 5, False) + wrapper.ggml_tensor_overhead() * num_tensors
            return graph_bytes
        
    def _allocate_inputs(self):
        # Set up inputs
        ctx_size = wrapper.ggml_tensor_overhead() * len(self.parse_context.created_tensors)
        io_ctx = wrapper.GGMLContext(ctx_size, backend=self.backend)
        for tensor in self.parse_context.created_tensors:
            vl_tensor = io_ctx.add_tensor_with_variable_shape(tensor.name, tensor.shape, tensor.type)
            vl_tensor.lowering_ctx = self.lowering_context

            vl_tensor.allocate()

        self.inputs_buffer = wrapper.ggml_backend_alloc_ctx_tensors(io_ctx.ctx, self.backend)
        if not self.inputs_buffer:
            raise RuntimeError("Failed to allocate buffer to store inputs!")
        wrapper.ggml_backend_buffer_clear(self.inputs_buffer, 0)

        return io_ctx

    def _build_forward(self, batch_params: BatchParams):
        """Build the GGM Graph from the parsed AST and ensure the tensors are loaded"""
        self.lowering_context = LoweringContext.from_parse_context(self.parse_context, batch_params)

        # Set up compute graph
        graph_size = (len(self.lowering_context.gguf_tensors) + len(self.lowering_context.graph)) * 5
        logging.debug(f"Creating GGML graph with {graph_size} nodes...")
        compute_ctx = self.ggml_ctx[GGMLContextType.COMPUTE]
        gf = wrapper.ggml_new_graph_custom(compute_ctx.ctx, graph_size, False)

        try:
            for node in self.parse_context.ast:
                produce_ggml_graph(self.lowering_context, compute_ctx.ctx, gf, node)
        except ParseError as exception:
            raise RuntimeError("Failed to parse") from exception

        output_tensor = self.lowering_context.graph.get("output")
        if not output_tensor:
            raise ValueError("Output tensor not found in the graph.")
        
        wrapper.ggml_set_name(output_tensor.ptr, b"result_output")
        
        logging.debug("Expanding graph...")
        wrapper.ggml_build_forward_expand(gf, output_tensor.ptr)
        # wrapper.ggml_graph_dump_dot(gf, ctypes.POINTER(wrapper.ggml_cgraph)(), b"graph.dot")

        self.compute_graph = gf
    
    def __call__(self, **kwargs: List[int | float]):
        """Run the GGML model with the provided input tensors"""
        if not self.compute_graph:
            # TODO: change this if statement to detect if the batch params have changed
            # and if so, re-build the graph
            self._build_forward()

        if not self.backend:
            raise RuntimeError("Cannot evaluate model without initializing the backend")

        for input_name in kwargs.keys():
            input_tensor = self.input_tensors[input_name]
            input_dtype = GGML_TYPE_TO_NUMPY_DTYPE[input_tensor.type]

            input_tensor.data = np.array(kwargs[input_name], dtype=input_dtype)

        backend_buffer_type = wrapper.ggml_backend_get_default_buffer_type(self.backend)
        if not backend_buffer_type:
            raise RuntimeError()
        
        allocr = wrapper.ggml_gallocr_new(backend_buffer_type)
        if not allocr:
            raise RuntimeError()
        
        wrapper.ggml_gallocr_alloc_graph(allocr, self.compute_graph)

        if self.backend_type == GGMLBackendType.CPU:
            logging.debug(f"Using {self.context_params.n_threads} threads for computation")
            wrapper.ggml_backend_cpu_set_n_threads(self.backend, self.context_params.n_threads)

        wrapper.ggml_backend_graph_compute(self.backend, self.compute_graph)

        output_ptr = wrapper.ggml_graph_get_tensor(self.compute_graph, b"result_output")
        assert output_ptr != ctypes.POINTER(wrapper.ggml_tensor)()

        output = Tensor.from_tensor_ptr("result_output", output_ptr)
        logprobs = output.data.reshape(
            -1, output.data.shape[0]
        ).copy()

        return logprobs

    def __repr__(self):
        return f"{self.__class__.__name__}({self.loaded_tensors.keys()})"