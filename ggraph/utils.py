
from __future__ import annotations
from typing import Optional, Dict, List, Union, TypeAlias, Any, TypeVar
from dataclasses import dataclass, field

import matplotlib
import matplotlib.pyplot as plt
from lark import Token
import numpy as np
from gguf.gguf_reader import ReaderField, ReaderTensor
from gguf.constants import Keys as GGUFKeys

from ggraph.wrapper import Tensor

matplotlib.use("agg")

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
class ContextParams:
    """Stores parameters related to the current GGML execution context"""
    # shard: Optional[Shard]
    n_threads: int
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
class BatchParams:
    """Stores parameters related to the current GGML batch"""
    n_tokens: int # batch size
    kv_output_pos: int # index of the first token in the batch

    def __eq__(self, other: object) -> bool:
        return (
            self.n_tokens == other.n_tokens and
            self.kv_output_pos == other.kv_output_pos
        ) if isinstance(other, BatchParams) else False

GraphArg: TypeAlias = Union[Tensor, int, str]

class ModelParams:
    """Provides an interface for consistently accessing model parameters from a GGUF file"""
    def __init__(self, params: Dict[str, ReaderField]):
        self._data = params
        self._arch = str(params[GGUFKeys.General.ARCHITECTURE].contents())

    def __getitem__(self, key: str) -> str:
        if "{arch}" in key:
            key = key.replace("{arch}", self._arch)
        return self._data[key].contents()
    
    T = TypeVar('T')
    def get(self, key: str, default: T = None) -> str | T:
        try:
            return self[key]
        except KeyError:
            return default
        
    @property
    def arch(self) -> str:
        return self._arch

    @property
    def n_ctx(self) -> int:
        return int(self[GGUFKeys.LLM.CONTEXT_LENGTH])
    
    @property
    def n_embd(self) -> int:
        return int(self[GGUFKeys.LLM.EMBEDDING_LENGTH])
    
    @property
    def n_layer(self) -> int:
        return int(self[GGUFKeys.LLM.BLOCK_COUNT])
    
    @property
    def n_expert(self) -> int:
        return int(self[GGUFKeys.LLM.EXPERT_COUNT])
    
    @property
    def n_expert_used(self) -> int:
        return int(self[GGUFKeys.LLM.EXPERT_USED_COUNT])
    
    @property
    def n_ffn(self) -> int:
        return int(self[GGUFKeys.LLM.FEED_FORWARD_LENGTH])
    
    @property
    def n_head(self) -> int:
        return int(self[GGUFKeys.Attention.HEAD_COUNT])
    
    @property
    def n_head_kv(self) -> int:
        return int(self.get(GGUFKeys.Attention.HEAD_COUNT_KV, self.n_head))
    
    @property
    def n_embd_k_gqa(self) -> int:
        return self.n_embd_head_k * self.n_head_kv
    
    @property
    def n_embd_v_gqa(self) -> int:
        return self.n_embd_head_v * self.n_head_kv
    
    @property
    def rope_finetuned(self) -> bool:
        return self.get(GGUFKeys.Rope.SCALING_FINETUNED) == "true"
    
    @property
    def rope_freq_base(self) -> float:
        return float(self.get(GGUFKeys.Rope.FREQ_BASE, 10000.0))
    
    @property
    def rope_freq_scale(self) -> float:
        return float(self.get(GGUFKeys.Rope.SCALING_FACTOR, 1.0))
    
    @property
    def rope_scaling_type(self) -> int:
        return int(self.get(GGUFKeys.Rope.SCALING_TYPE, 1))
    
    @property
    def rope_attn_factor(self) -> float:
        return float(self.get(GGUFKeys.Rope.SCALING_ATTN_FACTOR, 1.0))
    
    @property
    def n_embd_head_k(self) -> int:
        return int(self.get(GGUFKeys.Attention.KEY_LENGTH, self.n_embd // self.n_head))
    
    @property
    def n_embd_head_v(self) -> int:
        return int(self.get(GGUFKeys.Attention.VALUE_LENGTH, self.n_embd // self.n_head))
    
    @property
    def n_rot(self) -> int:
        return int(self.get(GGUFKeys.Rope.DIMENSION_COUNT, self.n_embd_head_k))
    
    @property
    def n_vocab(self) -> int:
        return int(self.get(GGUFKeys.LLM.VOCAB_SIZE, len(self._data[GGUFKeys.Tokenizer.LIST].parts)))
    
    @property
    def f_norm_rms_eps(self) -> float:
        return float(self[GGUFKeys.Attention.LAYERNORM_RMS_EPS])
    
    @property
    def stop_tokens(self) -> List[int]:
        stop_tokens = [
            self.get(GGUFKeys.Tokenizer.EOS_ID, ""),
            self.get(GGUFKeys.Tokenizer.EOT_ID, ""),
            self.get(GGUFKeys.Tokenizer.EOM_ID, ""),
            self.get(GGUFKeys.Tokenizer.PAD_ID, ""),
            self.get(GGUFKeys.Tokenizer.SEP_ID, ""),
            self.get(GGUFKeys.Tokenizer.UNK_ID, ""),
            self.get(GGUFKeys.Tokenizer.MASK_ID, ""),
        ]
        return [int(token) for token in stop_tokens if isinstance(token, int) or token.isdigit()]

    
    def to_default_ggml_context_params_dict(self) -> Dict[str, Any]:
        """Returns the default values for the context, provided in the gguf file"""
        return dict(
            n_ctx=self.n_ctx,
            rope_freq_base=self.rope_freq_base,
            rope_freq_scale=self.rope_freq_scale,
            rope_scaling_type=self.rope_scaling_type,
        )
    
def ensure_args(function_name: str, args: List[Tensor | int | float | str | None], types: List[type], token: Token) -> None:
    if len(args) != len(types):
        raise ParseError(f"Error in function call '{function_name}'. Expected {len(types)} arguments but got {len(args)}", token)
    
    for idx, (arg, expected_type) in enumerate(zip(args, types)):
        if not isinstance(arg, expected_type):
            raise ParseError(f"Error in function call '{function_name}'. Expected parameter {idx + 1} to be of type {expected_type.__name__}, but got {type(arg).__name__}", token)


def plot_logprob_heatmap(logprobs):
    plt.figure(figsize=(12, 6), dpi=300)
    plt.imshow(logprobs.T, aspect='auto', interpolation='nearest', cmap='viridis')
    # plt.matshow(logprobs.T)
    plt.colorbar(label='Logprob')
    plt.xlabel('Token Index')
    plt.ylabel('Vocab Index')
    plt.title('Logprob Heatmap')
    plt.tight_layout()
    plt.savefig('logprobs.png')

def plot_attention_heatmap(attn_weights: np.ndarray, layer):
    num_heads = attn_weights.shape[0]

    fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))
    for head_idx in range(num_heads):
        ax = axes[head_idx] if num_heads > 1 else axes
        im = ax.imshow(attn_weights[head_idx], cmap='Blues', interpolation='nearest')
        ax.set_title(f'Head {head_idx} - Layer {layer}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f'attention_{layer:02}.png')


def plot_attention_heatmap_avg(attn_weights: np.ndarray, layer):
    avg_attn = np.mean(attn_weights, axis=0)  # average over heads, shape: (seq_len, seq_len)
    plt.figure(figsize=(12, 6), dpi=300)
    plt.imshow(avg_attn, cmap='Blues', interpolation='nearest')
    plt.title(f'Average Attention - Layer {layer}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar(label='Attention Weight')
    plt.tight_layout()
    plt.savefig(f'attention_{layer:02}.png')
