import time
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np

from gglm.models import GGMLModel, GGMLBackendType, Tensor
from gglm.utils import plot_logprob_heatmap, plot_attention_heatmap, plot_attention_heatmap_avg
from gglm.wrapper import gen
from gglm.inference_engine import GGMLInferenceEngine
from transformers.models.qwen2 import Qwen2TokenizerFast

gguf_path = "/mnt/c/Users/salex/.cache/lm-studio/models/lmstudio-community/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"

np.set_printoptions(threshold=100000, linewidth=128)


n_ctx = 128
tokenizer: Qwen2TokenizerFast = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")

generated_output = []
start_time = time.time()

print("Generating output...")

def look_up_tensor(model: GGMLModel, name: str):
    if model.compute_graph is None:
        raise ValueError("model.compute_graph is None! The compute graph must be built before looking up tensors.")
    return Tensor.from_tensor_ptr(name, gen.ggml_graph_get_tensor(model.compute_graph, name.encode()))

try:
    inference_engine = GGMLInferenceEngine(gguf_path, tokenizer, n_ctx=n_ctx, n_threads=12)

    conversation_result = inference_engine.generate(input_conversation=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of England?"},
    ])

    print(conversation_result)

except KeyboardInterrupt:
    print("\n")
    logging.info("Sampling interrupted by user.")

# end_time = time.time()
# duration = end_time - start_time
# logging.info(f"Sampled {len(generated_output)} tokens in {duration:.2f} seconds ({len(generated_output) / duration:.2f} tok/sec)")