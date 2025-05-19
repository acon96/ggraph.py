import time
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np

from gglm.models import GGMLModel, GGMLBackendType, Tensor
from gglm.utils import plot_logprob_heatmap, plot_attention_heatmap
from gglm.wrapper import gen
from transformers.models.qwen2 import Qwen2TokenizerFast

gguf_path = "/mnt/c/Users/salex/.cache/lm-studio/models/lmstudio-community/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"

np.set_printoptions(threshold=100000, linewidth=128)

def pad(input_tokens: list[int | float], n_ctx: int, value: int | float):
    
    if isinstance(value, float):
        dtype = np.float32
    elif isinstance(value, int):
        dtype = np.int32
    else:
        raise ValueError()
    
    return np.array(input_tokens + [value] * (n_ctx - len(input_tokens)), dtype=dtype)

def causal_mask(n_ctx: int):
    """TODO: look at ggml.ggml_diag_mask_inf()"""
    mask = np.triu(np.ones((n_ctx, n_ctx), dtype=np.float32), k=1)
    mask = np.where(mask == 1, -np.inf, 0)
    return mask

def probs_from_logits(logits: np.ndarray, *, temperature: float):
    logits = np.asarray(logits) / temperature
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return probs

def top_k_tokens(probs: np.ndarray, top_k: int):
    """takes the top k probabilities and sets all others to zero"""
    if top_k <= 0:
        return probs
    indices = np.argpartition(-probs, top_k)[:top_k]
    mask = np.zeros_like(probs)
    mask[indices] = 1
    return probs * mask

def top_p_tokens(probs: np.ndarray, top_p: float):
    """only consideres tokens in the top p percentile of the distribution. sets all others to zero"""
    cumulative_sum = probs.cumsum(axis=-1, dtype=np.float32) <= top_p
    mask = np.zeros_like(probs)
    mask[cumulative_sum] = 1
    return probs * mask

def sample_from_logits(logits: np.ndarray, *, temperature: float, top_p: float, top_k: int):
    probs = probs_from_logits(logits, temperature=temperature)
    probs = top_p_tokens(probs, top_p)
    probs = top_k_tokens(probs, top_k)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.random.choice(len(probs), p=probs)

n_ctx = 6
tokenizer: Qwen2TokenizerFast = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")

model = GGMLModel(gguf_path, n_ctx=n_ctx)
model._build_forward()

print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id} (type: {type(tokenizer.pad_token_id)})")
if tokenizer.pad_token_id is None:
    raise ValueError("tokenizer.pad_token_id is None! Please check the tokenizer setup.")

chat_template_kv = model.model_params._data["tokenizer.chat_template"]
tokenizer.chat_template = chat_template_kv.contents()

# input_ids: list[int] = tokenizer.apply_chat_template(
#     conversation=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the capital of England?"},
#     ],
#     add_generation_prompt=True,
# )


input_ids = tokenizer(text="Today I want to ").data["input_ids"]
logging.debug(f"{input_ids=}")

generated_output = []
start_time = time.time()

print("Generated output: ", end="", flush=True)
print()

def look_up_tensor(name: str):
    if model.compute_graph is None:
        raise ValueError("model.compute_graph is None! The compute graph must be built before looking up tensors.")
    return Tensor.from_tensor_ptr(name, gen.ggml_graph_get_tensor(model.compute_graph, name.encode()))

try:
    for i in range(2):

        padded_input = pad(input_ids, n_ctx=n_ctx, value=tokenizer.pad_token_id).tolist()
        input_positions = [float(x) for x in range(n_ctx)]
        kq_mask = causal_mask(n_ctx=n_ctx).flatten().tolist()

        result = model(
            input_tokens=padded_input,
            input_positions=input_positions,
            kq_mask=kq_mask,
        )

        output_pos = len(input_ids) - 1
        logits = result.T[output_pos]

        # get the top 10 most likely tokens
        # top_10_tokens = np.argpartition(logits, -10)[-10:]
        # print("Top 10 tokens: ", end="")
        # for token in top_10_tokens:
        #     print(f"{tokenizer.decode([token])} ({token})", end=", ")
        # print()

        plot_logprob_heatmap(
            logprobs=result
        )

        for layer in [0]:
            kq = look_up_tensor(f"kq_{layer}")
            print(f"{kq=} {kq.data=}")

            kq_softmax = look_up_tensor(f"kq_softmax_{layer}")
            print(f"{kq_softmax=} {kq_softmax.data=}")
            plot_attention_heatmap(kq.data, layer)

        # print(f"{result=} {result.shape=} {result.dtype=}")

        # kq_mask_tensor = look_up_tensor("kq_mask")
        # print(f"{kq_mask_tensor=} {kq_mask_tensor.data}")

        new_token = sample_from_logits(logits=logits, temperature=0.7, top_p=0.95, top_k=40)
        decoded = tokenizer.decode([new_token])
        input_ids.append(new_token)
        generated_output.append(new_token)

        print(decoded, end="", flush=True)

        input("Press enter to continue...")
    print("\n")
except KeyboardInterrupt:
    print("\n")
    logging.info("Sampling interrupted by user.")

end_time = time.time()
duration = end_time - start_time
logging.info(f"Sampled {len(generated_output)} tokens in {duration:.2f} seconds ({len(generated_output) / duration:.2f} tok/sec)")