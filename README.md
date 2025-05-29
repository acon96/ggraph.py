# ggraph: Python Inference for GGML Models

`ggraph` provides a Python interface and Graph Definition Language for running GGML-based model inference. It enables running and experimenting with GGML models (such as Llama, Qwen, etc.) directly from Python, while also enabling easy distribution of models via a KV entry in a model's GGUF file.

## Features
- Custom Graph Definition Language for defining GGML Models
- Load and run GGUF models from Python

## Installation

Install the package from PyPI:

```bash
pip install ggraph
```

You may also need to install system dependencies for the GGML backend (see [llama.cpp](https://github.com/ggml-org/llama.cpp) for details).

## Usage Example

Below is a minimal example of running inference with a GGML model and a HuggingFace tokenizer:

```python
from gglm.inference_engine import GGMLInferenceEngine
from transformers.models.qwen2 import Qwen2TokenizerFast

gguf_path = "/path/to/model.gguf"  # Path to your GGML model file
n_ctx = 256
tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")

inference_engine = GGMLInferenceEngine(gguf_path, tokenizer, n_ctx=n_ctx, n_threads=12)
result = inference_engine.generate(input_conversation=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of England?"},
])
print(result.generated_text)
```

## GGML Python Bindings
`ggraph` uses a custom set of bindings generated directly from the GGML/Llama.cpp source code using a modified custom fork of ctypeslib that uses clang to generate the bindings. That fork has then been further modified to generate the wrapper for this project. The modified clang2py can be found here: https://github.com/acon96/ctypeslib-ggml

## Project Structure

- `gglm/` - Core Python package
  - `inference_engine.py` - Main inference logic
  - `sharded_inference_engine.py` - Sharded inference support
  - `models/` - Model graph definitions and utilities
  - `wrapper/` - Low-level bindings to GGML C/C++ libraries
- `scripts/` - Helper scripts for configuration and binding generation

## License

MIT License. See [LICENSE](LICENSE) for details.
