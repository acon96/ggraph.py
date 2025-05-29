import argparse
import time
import logging
import os

from gglm.models import GGMLBackendType
from gglm.inference_engine import GGMLInferenceEngine
from transformers.models.qwen2 import Qwen2TokenizerFast

def main():
    parser = argparse.ArgumentParser(description="Run inference with a GGUF model.")
    parser.add_argument("--model", "-m", type=str, help="Path to the GGUF model file.", required=True)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--backend", "-b", type=str, choices=["cpu", "cuda", "rocm"], default="cpu",
                        help="Backend to use for inference (default: cpu).")
    parser.add_argument("--n_ctx", "-c", type=int, default=256, help="Context size for the model (default: 256).")
    parser.add_argument("--n_threads", "-t", type=int, default=os.cpu_count(), help="Number of threads to use for inference (default: all cores).")
    parser.add_argument("--n_predict", "-p", type=int, default=128, help="Number of tokens to predict (default: 128).")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (default: 0.8).")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter (default: 40).")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter (default: 0.95).")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    start_time = time.time()
    print("Generating output...")

    try:
        inference_engine = GGMLInferenceEngine(args.model, n_ctx=args.n_ctx, n_threads=args.n_threads)

        result = inference_engine.generate(input_conversation=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of England?"},
        ])

        print(result.generated_text)

        end_time = time.time()
        duration = end_time - start_time
        tps = result.num_generated_tokens / duration
        logging.info(f"Sampled {result.num_generated_tokens} tokens in {duration:.2f} seconds ({tps:.2f} tok/sec)")

    except KeyboardInterrupt:
        print("\n")
        logging.info("Sampling interrupted by user.")


if __name__ == "__main__":
    main()