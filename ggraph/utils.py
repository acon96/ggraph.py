
from __future__ import annotations
from typing import Optional, Dict, List, Union, TypeAlias, Any, TypeVar, Mapping, Tuple
from dataclasses import dataclass, field
from functools import cache, cached_property
from datetime import datetime
import logging
import json

import jinja2
from jinja2.sandbox import SandboxedEnvironment
from jinja2.ext import loopcontrols
import matplotlib
import matplotlib.pyplot as plt
from lark import Token
import numpy as np
from gguf.gguf_reader import ReaderField, ReaderTensor
from gguf.constants import Keys as GGUFKeys, RopeScalingType, TokenType
from tokenizers import Tokenizer, Regex, SplitDelimiterBehavior, Encoding, AddedToken
from tokenizers import pre_tokenizers, decoders, processors, normalizers, models

from ggraph.wrapper import Tensor
from ggraph.wrapper.gen import LLAMA_ROPE_SCALING_TYPE_NONE, LLAMA_ROPE_SCALING_TYPE_LINEAR, LLAMA_ROPE_SCALING_TYPE_YARN, LLAMA_ROPE_SCALING_TYPE_LONGROPE, LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED

matplotlib.use("agg")

logger = logging.getLogger(__name__)

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

def build_pre_tokenizer_patterns(patterns: Dict[str, List[str]], aliases: Dict[str, str]) -> Mapping[str, pre_tokenizers.PreTokenizer]:
    """
    Builds a dictionary of pre-tokenizer patterns from a dictionary of string patterns.
    """
    result = {
        key: pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=Regex(pattern), behavior=SplitDelimiterBehavior.ISOLATED.value) for pattern in patterns
        ] + [pre_tokenizers.ByteLevel()]) for key, patterns in patterns.items()
    }
    # Add aliases to the base patterns
    for alias, base in aliases.items():
        if alias in result:
            raise ValueError(f"Alias '{alias}' conflicts with an existing pattern.")
        result[alias] = result[base]
    return result

PRE_TOKENIZER_PATTERNS = build_pre_tokenizer_patterns(
    {
        "default": [r"[\\p{P}\\$\\+<=>\\^~\\|]+",
                    r"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    r"\\p{N}+",
                    r"[0-9][0-9][0-9]"],
        "llama3": [r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"],
        "gpt2": [r"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"],
        "qwen2": [r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",]
    },
    {
        "stablelm2": "qwen2",
        "deepseek-r1-qwen": "qwen2",
        "llama-v3": "llama3",
        "llama-bpe": "llama3",
    }
)

def ggml_bpe_tokenizer(
    vocab: Union[str, Dict[str, int]],
    merges: Union[str, List[Tuple[str, str]]],
    pre_tokenizer: str = "default",
) -> Tokenizer:
    """Creates a Byte-level BPE tokenizer similar to the GPT-2 tokenizer"""
    tokenizer = Tokenizer(models.BPE(vocab, merges))

    tokenizer.pre_tokenizer = PRE_TOKENIZER_PATTERNS.get(pre_tokenizer, PRE_TOKENIZER_PATTERNS["default"])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel()

    return tokenizer

def ggml_spm_tokenizer(
    vocab: Union[str, Dict[str, int]],
    merges: Union[str, List[Tuple[str, str]]]
) -> Tokenizer:
    """Creates a SentencePiece BPE tokenizer with byte fallback similar to the Llama tokenizer"""
    tokenizer = Tokenizer(models.BPE(vocab, merges, unk_token="<unk>", byte_fallback=True))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")
])
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(content=" ", left=1, right=0)
    ])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A",
        pair="<s> $A <s> $B",
        special_tokens=[("<s>", 1)],
    )

    return tokenizer

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
        match self.get(GGUFKeys.Rope.SCALING_TYPE, "").upper():
            case "":
                return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
            case RopeScalingType.NONE.name:
                return LLAMA_ROPE_SCALING_TYPE_NONE
            case RopeScalingType.LINEAR.name:
                return LLAMA_ROPE_SCALING_TYPE_LINEAR
            case RopeScalingType.YARN.name:
                return LLAMA_ROPE_SCALING_TYPE_YARN
            case RopeScalingType.LONGROPE.name:
                return LLAMA_ROPE_SCALING_TYPE_LONGROPE
            case _:
                raise ParseError(f"Unknown rope scaling type: {self.get(GGUFKeys.Rope.SCALING_TYPE)}", None)
    
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
            self.get(GGUFKeys.Tokenizer.EOS_ID, -1),
            self.get(GGUFKeys.Tokenizer.EOT_ID, -1),
            self.get(GGUFKeys.Tokenizer.EOM_ID, -1),
            self.get(GGUFKeys.Tokenizer.PAD_ID, -1),
            self.get(GGUFKeys.Tokenizer.SEP_ID, -1),
            self.get(GGUFKeys.Tokenizer.UNK_ID, -1),
            self.get(GGUFKeys.Tokenizer.MASK_ID, -1),
        ]
        return [int(token) for token in stop_tokens if isinstance(token, int) or token.isdigit() and int(token) >= 0]
    
    @property
    def special_tokens(self) -> Dict[str, str]:
        special_tokens_map: Dict[str, int | str] = {
            "bos_token": self.get(GGUFKeys.Tokenizer.BOS_ID, -1),
            "eos_token": self.get(GGUFKeys.Tokenizer.EOS_ID, -1),
            "unk_token": self.get(GGUFKeys.Tokenizer.UNK_ID, self.get(GGUFKeys.Tokenizer.PAD_ID, 0)),
            "sep_token": self.get(GGUFKeys.Tokenizer.SEP_ID, -1),
            "pad_token": self.get(GGUFKeys.Tokenizer.PAD_ID, -1),
            "mask_token": self.get(GGUFKeys.Tokenizer.MASK_ID, -1),
        }

        token_list = self.get(GGUFKeys.Tokenizer.LIST, [])
        special_tokens_map = { k: token_list[v] for k, v in special_tokens_map.items() if v >= 0 }
        return special_tokens_map
    
    @cached_property
    def tokenizer(self) -> Tokenizer:
        logger.debug("Building tokenizer...")

        tokenizer_type = self.get(GGUFKeys.Tokenizer.MODEL, "no_vocab")
        token_list = self.get(GGUFKeys.Tokenizer.LIST, [])
        token_types = self.get(GGUFKeys.Tokenizer.TOKEN_TYPE, [])

        if "phi-3" in self.arch.lower() or "phi3" in self.arch.lower():
            token_overrides = {
                "<unk>": { "rstrip": False },
                "</s>": { "rstrip": True },
                "<unk>": { "rstrip": False },
                "<s>": { "rstrip": False },
                "<|endoftext|>": { "rstrip": False },
            }
            rstrip_special_tokens = True
        else:
            token_overrides = {}
            rstrip_special_tokens = False

        base_tokens = {}
        added_tokens = []
        special_tokens = []
        for idx, (token, token_type) in enumerate(zip(token_list, token_types)):
            if token_type == TokenType.NORMAL or token_type == TokenType.BYTE:
                base_tokens[token] = idx
            elif token_type == TokenType.USER_DEFINED or token_type == TokenType.CONTROL:
                add_to = special_tokens if token_type == TokenType.CONTROL else added_tokens
                is_special_token = token_type == TokenType.CONTROL
                token_kwargs = dict(lstrip=False, rstrip=rstrip_special_tokens and is_special_token, normalized=False, special=is_special_token)
                token_kwargs.update(token_overrides.get(token, {}))
                add_to.append(AddedToken(content=token, **token_kwargs))
            
        merges = [ tuple(merge.split(" ")) for merge in self.get(GGUFKeys.Tokenizer.MERGES, []) if len(merge.split(" ")) == 2]

        if tokenizer_type == "gpt2":
            pre_tokenizer = self.get(GGUFKeys.Tokenizer.PRE, None)
            tokenizer = ggml_bpe_tokenizer(vocab=base_tokens, merges=merges, pre_tokenizer=pre_tokenizer)
        elif tokenizer_type == "llama":
            tokenizer = ggml_spm_tokenizer(vocab=base_tokens, merges=merges)
        # elif tokenizer_type == "bert":
        #     tokenizer = Tokenizer(WordPiece(vocab=base_tokens, unk_token=unk_token, max_input_chars_per_word=100))
        # elif tokenizer_type == "t5":
        #     tokenizer = Tokenizer(Unigram(vocab=base_tokens, unk_id=unk_token, byte_fallback=True))
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.add_tokens(added_tokens)
        
        return tokenizer

    
    def to_default_ggml_context_params_dict(self) -> Dict[str, Any]:
        """Returns the default values for the context, provided in the gguf file"""
        return dict(
            n_ctx=self.n_ctx,
            rope_freq_base=self.rope_freq_base,
            rope_freq_scale=self.rope_freq_scale,
            rope_scaling_type=self.rope_scaling_type,
        )

@cache
def compile_jinja_template(template: str) -> jinja2.Template:
    """
    Compiles a Jinja2 template string into a Template object.
    
    Args:
        template: The Jinja2 template string to compile.
        
    Returns:
        A compiled Jinja2 Template object.
    """
    env = SandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # We override the built-in tojson filter because Jinja's default filter escapes HTML characters
        # We also expose some options like custom indents and separators
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(format):
        return datetime.now().strftime(format)
    
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    
    return env.from_string(template)

def render_chat_with_jinja(
        tokenizer: Tokenizer, 
        conversation: List[Dict[str, str]], 
        chat_template: str, 
        tools: Optional[List[Dict]] = None,
        documents: Optional[List[Dict]] = None,
        add_generation_prompt: bool = True,
        chat_template_kwargs: Optional[Dict[str, Any]] = None) -> List[int]:
    """
    Renders a conversation using a jinja2 chat template, returning the tokenized input IDs.
    
    Args:
        tokenizer: The tokenizer to use for encoding.
        conversation: A list of dictionaries representing the conversation.
        chat_template: The Jinja2 template string to apply to the conversation.
        tools: Optional list of tools to include in the conversation.
        documents: Optional list of documents to include in the conversation.
        add_generation_prompt: Whether to add a generation prompt at the end.
        chat_template_kwargs: Additional keyword arguments to pass to the template.
        
    Returns:
        A list of token IDs representing the conversation.
    """
    jinja_template = compile_jinja_template(chat_template)    

    rendered_chat = jinja_template.render(
        messages=conversation,
        tools=tools,
        documents=documents,
        add_generation_prompt=add_generation_prompt,
        **(chat_template_kwargs or {})
    )

    logger.debug(f"Rendered chat template: {rendered_chat}")

    encoding: Encoding = tokenizer.encode(rendered_chat)
    return encoding.ids

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
