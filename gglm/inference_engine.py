from typing import Optional, Protocol, Any
from dataclasses import dataclass
import logging
import numpy as np
from gglm.models import GGMLModel
from gglm.utils import plot_logprob_heatmap, plot_attention_heatmap, plot_attention_heatmap_avg

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
    # probs = top_p_tokens(probs, top_p)
    # probs = top_k_tokens(probs, top_k)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.random.choice(len(probs), p=probs)

class TokenizerProtocol(Protocol):
    def __call__(self, text: str) -> dict[str, list[int]]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...

    def apply_chat_template(self, conversation: list[dict[str, str]], add_generation_prompt: bool) -> dict[str, list[int]]:
        ...

    @property
    def pad_token_id(self) -> int:
        ...

    @property
    def chat_template(self) -> dict[str, str]:
        ...

    @chat_template.setter
    def chat_template(self, value: dict[str, str]):
        ...

@dataclass(kw_only=True)
class InferenceResult:
    """Result of the inference."""
    generated_tokens: list[int]
    num_generated_tokens: int
    generated_text: str
    conversation: Optional[list[dict[str, str]]] = None

class GGMLInferenceEngine:
    """Inference engine for GGML models."""

    model: GGMLModel
    tokenizer: Optional[TokenizerProtocol]

    def __init__(self, gguf_path: str, tokenizer: Optional[TokenizerProtocol] = None,  n_ctx: int = 32, **kwargs):
        self.model = GGMLModel(gguf_path, n_ctx=n_ctx, **kwargs)
        self.tokenizer = tokenizer

        chat_template_kv = self.model.model_params._data["tokenizer.chat_template"]
        tokenizer.chat_template = chat_template_kv.contents()

    def generate(self, *, input_prompt: Optional[str] = None, input_tokens: Optional[list[int]] = None, input_conversation: Optional[list[dict[str, str]]] = None) -> InferenceResult:
        """Generates text from the input prompt or tokens."""
        n_ctx = self.model.context_params.n_ctx
        stop_tokens = self.model.model_params.stop_tokens

        if input_prompt is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not set.")
            input_tokens = self.tokenizer(text=input_prompt).data["input_ids"]

        if input_conversation is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not set.")
            input_tokens = self.tokenizer.apply_chat_template(input_conversation, add_generation_prompt=True)
        
        if input_tokens is None:
            raise ValueError("No input tokens provided.")
        
        input_ids = input_tokens[:n_ctx]

        # process the prompt
        n_tokens = len(input_ids)
        kq_mask = causal_mask(n_ctx=n_ctx)
       
        logging.debug(f"Processing {len(input_ids)} input tokens with n_ctx={n_ctx}")
        result = self.model(
            input_tokens=input_ids,
            input_positions=[float(x) for x in range(n_tokens)],
            kq_mask=kq_mask[:n_tokens].flatten().tolist(),
            kv_output_pos=0
        )

        logits = result[n_tokens - 1]
        
        last_output = sample_from_logits(logits=logits, temperature=0.7, top_p=0.95, top_k=40)
        outputs = [last_output]
        # logging.debug(f"Generated token: {last_output} ({self.tokenizer.decode(outputs)})")

        # process next tokens one by one
        while len(input_ids) + len(outputs) < n_ctx and last_output not in stop_tokens:
            cur_pos = len(input_ids) + len(outputs) - 1
            result = self.model(
                input_tokens=[last_output],
                input_positions=[cur_pos],
                kq_mask=kq_mask[cur_pos],
                kv_output_pos=cur_pos,
            )

            logits = result[0]

            last_output = sample_from_logits(logits=logits, temperature=0.7, top_p=0.95, top_k=40)
            outputs.append(last_output)
            # logging.debug(f"Generated token: {last_output} ({self.tokenizer.decode([last_output])})")

        if input_conversation is not None:
            output_text = self.tokenizer.decode(outputs)
            output_conversation = input_conversation.copy()
            output_conversation.append({"role": "assistant", "content": output_text})
            return InferenceResult(generated_tokens=outputs, num_generated_tokens=len(outputs), generated_text=output_text, conversation=output_conversation)
        
        output_text = self.tokenizer.decode(outputs)
        return InferenceResult(generated_tokens=outputs, num_generated_tokens=len(outputs), generated_text=output_text)