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
from ggraph.inference_engine import GGMLInferenceEngine

gguf_path = "/path/to/model.gguf"  # Path to your GGML model file
n_ctx = 256

inference_engine = GGMLInferenceEngine(gguf_path, n_ctx=n_ctx, n_threads=12)
result = inference_engine.generate(input_conversation=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of England?"},
])
print(result.generated_text)
```

### CLI
You can also run supported GGUF Models directly from the command line by using the `ggraph-cli` command. The usage and an example invocation is below:

```
usage: ggraph-cli [-h] --model MODEL [--debug] [--backend {cpu,cuda,rocm}] [--n_ctx N_CTX] [--n_threads N_THREADS]
                  [--n_predict N_PREDICT] [--stream] [--temperature TEMPERATURE] [--top_k TOP_K] [--top_p TOP_P]
                  [--conversation-system CONVERSATION_SYSTEM] [--conversation-user CONVERSATION_USER] [--interactive]

ggraph-cli -m "./models/Qwen2.5-0.5B-Instruct-Q6_K.gguf" --n_ctx 1024 --n_predict 128 --interactive --stream
```

> NOTE: Not all of the CLI flags work as of right now. They are reserved for future use.

## GGML Python Bindings
`ggraph` uses a custom set of bindings generated directly from the GGML/Llama.cpp source code using a modified custom fork of ctypeslib that uses clang to generate the bindings. That fork has then been further modified to generate the wrapper for this project. The modified clang2py can be found here: https://github.com/acon96/ctypeslib-ggml

## Project Structure

- `ggraph/` - Core Python package
  - `inference_engine.py` - Main inference logic
  - `sharded_inference_engine.py` - Sharded inference support
  - `models/` - Model graph definitions and utilities
  - `wrapper/` - Low-level bindings to GGML C/C++ libraries
- `scripts/` - Helper scripts for configuration and binding generation

## License

MIT License. See [LICENSE](LICENSE) for details.
