
import argparse

from knowledge_assistant.llm_client import LocalLLM

def main():
    parser = argparse.ArgumentParser(
        description="Enterprise AI Knowledge Assistant"
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Question to ask the local LLM",
    )

    parser.add_argument(
        "--model",
        default="qwen2.5:1.5b",
        help="Ollama model name",
    )

    args = parser.parse_args()

    assistant = LocalLLM(model=args.model)

    print("\nGenerating response...\n")

    answer = assistant.generate(args.prompt)

    print("Assistant:")
    print(answer)

if __name__ == "__main__":
    main()