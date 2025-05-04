import time
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import ggml

from gglm.models import GGMLModel, GGMLBackendType, Tensor
from gglm.utils import get_tensor_to_numpy
from transformers.models.qwen2 import Qwen2TokenizerFast

gguf_path = "/mnt/c/Users/salex/.cache/lm-studio/models/lmstudio-community/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"

def pad(input_tokens: list[int | float], n_ctx: int, value: int | float):
    return input_tokens + [value] * (n_ctx - len(input_tokens))

def causal_mask(n_ctx: int):
    return [[float(0.0 if i <= j else '-inf') for i in range(n_ctx)] for j in range(n_ctx)]

def sample_from_logits(logits: np.ndarray, *, temperature: float, top_p: float, top_k: float):
    logits = np.asarray(logits) / temperature
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.random.choice(len(probs), p=probs)

n_ctx = 256
model = GGMLModel(gguf_path, n_ctx=n_ctx)
model._build_forward()

tokenizer: Qwen2TokenizerFast = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
chat_template_kv = model.model_params._data["tokenizer.chat_template"]
tokenizer.chat_template = chat_template_kv.contents()

input_ids: list[int] = tokenizer.apply_chat_template(
    conversation=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of England?"},
    ],
    add_generation_prompt=True,
)

generated_output = []
start_time = time.time()

print("Generated output: ", end="", flush=True)

def look_up_tensor(name: str):
    return Tensor.from_tensor_ptr(name, ggml.ggml_graph_get_tensor(model.compute_graph, name.encode()))

try:
    for i in range(32):
        result = model(
            input_tokens=pad(input_ids, n_ctx=n_ctx, value=tokenizer.pad_token_id),
            input_positions=list(range(n_ctx)),
            kq_mask=causal_mask(n_ctx=n_ctx),
        )

        # print(get_tensor_to_numpy(look_up_tensor("get_rows_aa2b")))

        output_pos = len(input_ids)
        logits = result.reshape(n_ctx, -1)[output_pos]

        new_token = sample_from_logits(logits=logits, temperature=0.7, top_p=0.95, top_k=40)
        input_ids.append(new_token)
        generated_output.append(new_token)

        print(tokenizer.decode([new_token]), end="", flush=True)
    print("\n")
except KeyboardInterrupt:
    print("\n")
    logging.info("Sampling interrupted by user.")

end_time = time.time()
duration = end_time - start_time
logging.info(f"Sampled {len(generated_output)} tokens in {duration:.2f} seconds ({len(generated_output) / duration:.2f} tok/sec)")