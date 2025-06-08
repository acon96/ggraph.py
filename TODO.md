# TODO
- [x] support token streaming
- [ ] support more models (llama, phi, qwen2.5, qwen3moe, etc.); need to load tokenizers from gguf to do this properly  
- [ ] support multiple ggml backends as part of the same graph; i.e. cpu + gpu at the same time or multiple gpus; also check if backends support a certain tensor type  
- [ ] make the pip packge install the ggml shared libraries from source so that the compiled version is as optimized as possible (`GGML_NATIVE=ON`)  
- [ ] support proper sampler chains
- [ ] support training  
- [ ] resolve unknown symbols in ggml files before loading all the tensors; this should be possible because we know the graph variables, parameter property names, and gguf tensor names immediately after parsing