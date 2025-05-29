# TODO
- [x] support token streaming
- [ ] support more models (llama, phi, qwen2.5, qwen3moe, etc.)  
- [ ] support multiple ggml backends as part of the same graph; i.e. cpu + gpu at the same time or multiple gpus  
- [ ] make the pip packge install the ggml shared libraries from source so that the compiled version is as optimized as possible (`GGML_NATIVE=ON`)  
- [ ] support proper sampler chains
- [ ] support training