import argparse
from pathlib import Path

from intent_classifier.classifier import HybridIntentClassifier, IntentClassifier
from intent_classifier.llm_parser import DEFAULT_LLM_MODEL, LLMIntentParser


DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "checkpoints" / "intent_classifier.pt"
)


def print_prediction(classifier: IntentClassifier, text: str) -> None:
    result = classifier.predict(text)
    print(f"text: {text}")
    print(f"intent: {result.intent.value}")
    print(f"target: {result.target}")
    print(f"confidence: {result.confidence:.3f}")
    print(f"source: {result.source}")
    if result.language is not None:
        print(f"language: {result.language}")
    if result.shelf_constraint is not None:
        print(f"shelf_constraint: {result.shelf_constraint}")
    if result.requested_detail is not None:
        print(f"requested_detail: {result.requested_detail}")
    if result.response_style is not None:
        print(f"response_style: {result.response_style}")
    if result.metadata and result.metadata.get("error"):
        print(f"error: {result.metadata['error']}")
    if result.metadata and result.metadata.get("hint"):
        print(f"hint: {result.metadata['hint']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run intent classifier prediction.")
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to classify. If omitted, starts an interactive prompt.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence before falling back to unknown.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--backend",
        choices=("bert", "llm", "hybrid"),
        default="bert",
        help="Intent parser backend. 'hybrid' uses BERT first and LLM fallback.",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="Ollama model used for --backend llm or --backend hybrid.",
    )
    parser.add_argument(
        "--llm-confidence-threshold",
        type=float,
        default=0.45,
        help="Minimum LLM confidence before returning unknown.",
    )
    parser.add_argument(
        "--llm-fallback-threshold",
        type=float,
        default=0.70,
        help="In hybrid mode, call the LLM when BERT confidence is below this value.",
    )
    return parser.parse_args()


class SimpleIntentClassifier(IntentClassifier):
    def __init__(self, confidence_threshold: float = 0.5, device: str = "auto"):
        # If device is None, IntentClassifier will automatically select the best available device (cuda, mps, or cpu).
        run_device = None
        if device != "auto":
            run_device = device
        super().__init__(checkpoint_path=DEFAULT_CHECKPOINT_PATH, confidence_threshold=confidence_threshold, device=run_device)

    def classify(self, text: str) -> tuple[str, str, float]:
        """Classify the intent of the given text and return the intent, target and confidence."""
        result = self.predict(text)
        if result.target is None:
            target = ""
        else:
            target = result.target
        return result.intent.value, target, result.confidence


def main() -> None:
    args = parse_args()
    bert_classifier = None
    if args.backend in {"bert", "hybrid"}:
        bert_classifier = IntentClassifier(
            checkpoint_path=args.checkpoint,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
        )

    if args.backend == "bert":
        classifier = bert_classifier
    elif args.backend == "llm":
        classifier = LLMIntentParser(
            model=args.llm_model,
            confidence_threshold=args.llm_confidence_threshold,
        )
    else:
        llm_parser = LLMIntentParser(
            model=args.llm_model,
            confidence_threshold=args.llm_confidence_threshold,
        )
        classifier = HybridIntentClassifier(
            bert_classifier=bert_classifier,
            llm_parser=llm_parser,
            llm_fallback_threshold=args.llm_fallback_threshold,
        )

    if args.text is not None:
        print_prediction(classifier, args.text)
        return

    print("Enter text to classify. Press Ctrl+C or submit an empty line to exit.")
    while True:
        text = input("> ").strip()
        if not text:
            break
        print_prediction(classifier, text)


if __name__ == "__main__":
    main()
