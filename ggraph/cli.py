import argparse
import time
import logging
import os

from ggraph.models import GGMLBackendType
from ggraph.inference_engine import GGMLInferenceEngine

logger: logging.Logger = None

def generate_assistant_conversation_turn(inference_engine: GGMLInferenceEngine, input_conversation: list[dict[str, str]], stream_output: bool):
    start_time = time.time()

    if stream_output:
        output_tokens = []
        stream = inference_engine.stream(input_conversation=input_conversation)
        for token in stream:
            print(token, end="", flush=True)
            output_tokens.append(token)
        print()

        num_tokens = len(output_tokens)
        result: str = "".join(output_tokens)
    else:
        output = inference_engine.generate(input_conversation=input_conversation)

        print(output.generated_text)
        num_tokens = output.num_generated_tokens
        result = output.generated_text

    end_time = time.time()
    duration = end_time - start_time
    tps = num_tokens / duration
    logger.info(f"Sampled {num_tokens} tokens in {duration:.2f} seconds ({tps:.2f} tok/sec)")

    return result

def main():
    global logger
    parser = argparse.ArgumentParser(description="Run inference with a GGUF model.")
    parser.add_argument("--model", "-m", type=str, help="Path to the GGUF model file.", required=True)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--backend", "-b", type=str, choices=["cpu", "cuda", "vulkan"], default="cpu",
                        help="Backend to use for inference (default: cpu).")
    parser.add_argument("--n_ctx", "-c", type=int, default=256, help="Context size for the model (default: 256).")
    parser.add_argument("--n_threads", "-t", type=int, default=os.cpu_count(), help="Number of threads to use for inference (default: all cores).")
    parser.add_argument("--n_predict", "-p", type=int, default=128, help="Number of tokens to predict (default: 128).")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (default: 0.8).")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter (default: 40).")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter (default: 0.95).")
    parser.add_argument("--conversation-system", type=str, default="You are a helpful assistant.",
                        help="System message for the conversation (default: 'You are a helpful assistant.').")
    parser.add_argument("--conversation-user", type=str, default="What is the capital of England?",
                        help="User message for the conversation (default: 'What is the capital of England?').")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode, allowing for continuous input and output.")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    input_conversation = [
        {"role": "system", "content": args.conversation_system},
    ]

    engine_kwargs = {}

    match args.backend:
        case "cpu":
            engine_kwargs["backend_type"] = GGMLBackendType.CPU
            engine_kwargs["n_threads"] = args.n_threads
        case "cuda":
            engine_kwargs["backend_type"] = GGMLBackendType.CUDA
        case "vulkan":
            engine_kwargs["backend_type"] = GGMLBackendType.VULKAN
        case _:
            raise ValueError(f"Invalid backend type: {args.backend}")

    try:
        inference_engine = GGMLInferenceEngine(args.model, n_ctx=args.n_ctx, **engine_kwargs)
        
        if args.interactive:
            logger.info("Entering interactive mode. Type 'exit' to quit.")
            while True:
                user_input = input("User: ")
                if user_input.lower() == "exit":
                    break
                input_conversation.append({"role": "user", "content": user_input})
                print("Assistant: ", end="")
                assistant_response = generate_assistant_conversation_turn(inference_engine, input_conversation, args.stream)
                input_conversation.append({"role": "assistant", "content": assistant_response})
        else:
            logger.info("Generating output...")
            input_conversation.append({"role": "user", "content": args.conversation_user})
            generate_assistant_conversation_turn(inference_engine, input_conversation, args.stream)

    except KeyboardInterrupt:
        print("\n")
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()