import asyncio
import numpy as np
from typing import Optional, List, Protocol
from concurrent.futures import ThreadPoolExecutor
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader

import ggml

from .models import GGMLModel

class TokenizerProtocol(Protocol):
    def tokenize(self, text: str) -> List[int]:
        ...

    def detokenize(self, tokens: List[int]) -> str:
        ...

class GGMLDynamicShardInferenceEngine(InferenceEngine):

    shard: Shard
    model: GGMLModel
    tokenizer: TokenizerProtocol
    shard_downloader: ShardDownloader
    _ggml_thread: ThreadPoolExecutor
    _tokenizer_thread: ThreadPoolExecutor
    _shard_lock: asyncio.Lock

    def __init__(self, shard_downloader: ShardDownloader):
        self.shard = None
        self.model = None
        self.tokenizer = None
        self.shard_downloader = shard_downloader

        self._ggml_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._tokenizer_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenizer")
        self._shard_lock = asyncio.Lock()

    async def ensure_shard(self, shard: Shard):
        if self.shard == shard:
            return

        async with self._shard_lock:
            model_path = await self.shard_downloader.ensure_shard(shard)
            self.shard = shard

            # TODO: init the tokenizer and model


    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        await self.ensure_shard(shard)
        tokens = await asyncio.get_running_loop().run_in_executor(self._tokenizer_thread, self.tokenizer.tokenize, prompt)
        return np.asarray(tokens)

    async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
        await self.ensure_shard(shard)
        return await asyncio.get_running_loop().run_in_executor(self._tokenizer_thread, self.tokenizer.tokenize, tokens)

    async def sample(self, x: np.ndarray) -> np.ndarray:
        pass

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
        await self.ensure_shard(shard)
        output_data = await asyncio.get_running_loop().run_in_executor(
            self._ggml_thread,
            lambda: np.array(self.model(input_embd=input_data))
        )

        return output_data, inference_state
    
    async def save_checkpoint(self, shard: Shard, path: str):
        await self.ensure_shard(shard)
        await asyncio.get_running_loop().run_in_executor(self._ggml_thread, lambda: self.model.save_weights(path))

    async def load_checkpoint(self, shard: Shard, path: str):
        await self.ensure_shard(shard)
        await asyncio.get_running_loop().run_in_executor(self._ggml_thread, lambda: self.model.load_weights(path))