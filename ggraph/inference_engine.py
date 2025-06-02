from typing import Optional, Generator, Any
from dataclasses import dataclass
import logging
import numpy as np
from ggraph.models import GGMLModel
from tokenizers import Tokenizer, Encoding

from ggraph.utils import render_chat_with_jinja

logger = logging.getLogger(__name__)

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
    tokenizer: Optional[Tokenizer]
    chat_template: Optional[str] = None

    def __init__(self, gguf_path: str, n_ctx: int = 32, **kwargs):
        self.model = GGMLModel(gguf_path, n_ctx=n_ctx, **kwargs)
        
        # map and load tokenizer
        self.tokenizer = self.model.model_params.tokenizer

        if self.tokenizer:
            chat_template_kv = self.model.model_params._data["tokenizer.chat_template"]
            self.chat_template = chat_template_kv.contents()

    def _get_tokens(self, *,
                    input_prompt: Optional[str] = None,
                    input_conversation: Optional[list[dict[str, str]]] = None) -> list[int]:
        if input_prompt is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not set.")
            
            encoded: Encoding = self.tokenizer.encode(input_prompt)
            input_tokens = encoded.ids

        elif input_conversation is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not set.")
            if self.chat_template is None:
                raise ValueError("Chat template is not set.")
            
            chat_template_kwargs = { **self.model.model_params.special_tokens }
            
            input_tokens = render_chat_with_jinja(
                self.tokenizer, conversation=input_conversation, chat_template=self.chat_template,
                add_generation_prompt=True, chat_template_kwargs=chat_template_kwargs)
        else:
            raise ValueError("Either input_prompt or input_conversation must be provided.")

        return input_tokens
    
    def _process_prompt(self, input_ids: list[int], kq_mask: np.ndarray) -> int:
        # process the prompt
        n_tokens = len(input_ids)
       
        logger.debug(f"Processing {len(input_ids)} input tokens...")
        result = self.model(
            input_tokens=input_ids,
            input_positions=[float(x) for x in range(n_tokens)],
            kq_mask=kq_mask[:n_tokens].flatten().tolist(),
            kv_output_pos=0
        )

        logits = result[n_tokens - 1]
        
        return sample_from_logits(logits=logits, temperature=0.7, top_p=0.95, top_k=40)
    
    def _generate_token(self, last_output: int, cur_pos: int, kq_mask: np.ndarray) -> int:
        result = self.model(
            input_tokens=[last_output],
            input_positions=[cur_pos],
            kq_mask=kq_mask[cur_pos],
            kv_output_pos=cur_pos,
        )

        logits = result[0]

        return sample_from_logits(logits=logits, temperature=0.7, top_p=0.95, top_k=40)

    def stream(self, *, 
                     input_prompt: Optional[str] = None, 
                     input_conversation: Optional[list[dict[str, str]]] = None) -> Generator[str, None, None]:
        """Generates text from the input prompt or tokens in a streaming manner."""
        n_ctx = self.model.context_params.n_ctx
        stop_tokens = self.model.model_params.stop_tokens
        kq_mask = causal_mask(n_ctx=n_ctx)

        input_tokens = self._get_tokens(input_prompt=input_prompt, input_conversation=input_conversation)

        input_ids = input_tokens[:n_ctx]

        last_output = self._process_prompt(input_ids, kq_mask)
        outputs = [last_output]
        yield self.tokenizer.decode([last_output], skip_special_tokens=True)

        # process next tokens one by one
        while len(input_ids) + len(outputs) < n_ctx and last_output not in stop_tokens:
            cur_pos = len(input_ids) + len(outputs) - 1
            last_output = self._generate_token(last_output, cur_pos, kq_mask)
            outputs.append(last_output)
            yield self.tokenizer.decode([last_output], skip_special_tokens=True)


    def generate(self, *, 
                 input_prompt: Optional[str] = None, 
                 input_tokens: Optional[list[int]] = None, 
                 input_conversation: Optional[list[dict[str, str]]] = None) -> InferenceResult:
        """Generates text from the input prompt or tokens."""
        n_ctx = self.model.context_params.n_ctx
        stop_tokens = self.model.model_params.stop_tokens
        kq_mask = causal_mask(n_ctx=n_ctx)

        if input_tokens is None:
            input_tokens = self._get_tokens(input_prompt=input_prompt, input_conversation=input_conversation)
        
        input_ids = input_tokens[:n_ctx]

        last_output = self._process_prompt(input_ids, kq_mask)
        outputs = [last_output]

        # process next tokens one by one
        while len(input_ids) + len(outputs) < n_ctx and last_output not in stop_tokens:
            cur_pos = len(input_ids) + len(outputs) - 1
            last_output = self._generate_token(last_output, cur_pos, kq_mask)
            outputs.append(last_output)

        if input_conversation is not None:
            output_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
            output_conversation = input_conversation.copy()
            output_conversation.append({"role": "assistant", "content": output_text})
            return InferenceResult(generated_tokens=outputs, num_generated_tokens=len(outputs), generated_text=output_text, conversation=output_conversation)
        
        output_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return InferenceResult(generated_tokens=outputs, num_generated_tokens=len(outputs), generated_text=output_text)