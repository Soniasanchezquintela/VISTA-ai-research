import argparse
from pathlib import Path

from intent_classifier.classifier import IntentClassifier


DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "checkpoints" / "intent_classifier.pt"
)


def print_prediction(classifier: IntentClassifier, text: str) -> None:
    result = classifier.predict(text)
    print(f"text: {text}")
    print(f"intent: {result.intent.value}")
    print(f"target: {result.target}")
    print(f"confidence: {result.confidence:.3f}")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = IntentClassifier(
        checkpoint_path=args.checkpoint,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
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
