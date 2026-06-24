from __future__ import annotations

import json
import re
from typing import Any, Optional

from intent_classifier.rule_extractor import normalize_target
from intent_classifier.schema import Intent, IntentResult


DEFAULT_LLM_MODEL = "gemma3:4b"

INTENT_DESCRIPTIONS = {
    Intent.DESCRIBE_SCENE: "The user wants a summary of the visible shelf or scene.",
    Intent.DESCRIBE_POINTED_PRODUCT: "The user wants information about the product they are pointing at.",
    Intent.NAVIGATE_TO_TARGET: "The user wants help finding or reaching a target product.",
    Intent.CONFIRM_TARGET_PRESENT: "The user asks whether a target product is present.",
    Intent.GET_PRICE: "The user asks for the price of a visible or target product.",
    Intent.READ_TEXT: "The user wants text on a product, label, sign, or package read aloud.",
    Intent.UNKNOWN: "The request is unrelated, unclear, or unsupported.",
}

ALLOWED_SHELF_CONSTRAINTS = {
    "top",
    "middle",
    "bottom",
    "upper",
    "lower",
    "left",
    "center",
    "right",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
}


class LLMIntentParser:
    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        confidence_threshold: float = 0.45,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature

    def parse(self, text: str) -> IntentResult:
        if not text.strip():
            return IntentResult(intent=Intent.UNKNOWN, target=None, confidence=0.0, source="llm")

        try:
            from ollama import chat
        except ImportError as exc:
            raise ImportError(
                "LLMIntentParser requires the 'ollama' package. "
                "Install project requirements before using the LLM parser."
            ) from exc

        try:
            response = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": text},
                ],
                options={"temperature": self.temperature},
            )
        except Exception as exc:
            return IntentResult(
                intent=Intent.UNKNOWN,
                target=None,
                confidence=0.0,
                source="llm",
                metadata={
                    "input_text": text,
                    "error": str(exc),
                    "hint": "Check that Ollama is installed, running, and has the selected model available.",
                },
            )

        content = response["message"]["content"]
        parsed = parse_json_object(content)
        return self._to_intent_result(text, content, parsed)

    def predict(self, text: str) -> IntentResult:
        return self.parse(text)

    def predict_batch(self, texts) -> list[IntentResult]:
        return [self.predict(text) for text in texts]

    def _system_prompt(self) -> str:
        intents = "\n".join(
            f"- {intent.value}: {description}"
            for intent, description in INTENT_DESCRIPTIONS.items()
        )
        shelf_constraints = ", ".join(sorted(ALLOWED_SHELF_CONSTRAINTS))

        return f"""
You parse natural supermarket assistant requests into strict JSON.

Allowed intents:
{intents}

Return only one JSON object with these keys:
- intent: one allowed intent string.
- target: product or object name, or null.
- confidence: number from 0.0 to 1.0.
- language: short language code like "en", "es", or null.
- shelf_constraint: one of [{shelf_constraints}], or null.
- requested_detail: short string such as "location", "price", "ingredients", "brand", "text", or null.
- response_style: short style hint such as "brief", "descriptive", "confirming", or null.

Rules:
- Use navigate_to_target when the user wants to find, locate, reach, or be guided to a product.
- Use confirm_target_present when the user asks if a product exists or is visible.
- Use describe_scene when the user asks what is around, on the shelf, or in front of them.
- Use describe_pointed_product when they ask about this product, that one, or the item they point at.
- Use get_price for price questions.
- Use read_text for reading labels, signs, ingredients, nutrition, or package text.
- Extract target in the user's words, without articles like "the", "el", or "la".
- Set shelf_constraint only when the user explicitly mentions a shelf or area such as top, lower, left, or right.
- Words like "somewhere", "anywhere", "around here", or "in this supermarket" are not shelf constraints.
- If the request is ambiguous, use unknown with low confidence.

Examples:
User: "Do you see horchata on the lower shelf?"
JSON: {{"intent":"confirm_target_present","target":"horchata","confidence":0.95,"language":"en","shelf_constraint":"lower","requested_detail":"presence","response_style":"brief"}}

User: "I am lost in this supermarket. Is there horchata somewhere? Can you specify the position?"
JSON: {{"intent":"confirm_target_present","target":"horchata","confidence":0.95,"language":"en","shelf_constraint":null,"requested_detail":"position","response_style":"brief"}}

User: "Where can I find coconut milk?"
JSON: {{"intent":"navigate_to_target","target":"coconut milk","confidence":0.95,"language":"en","shelf_constraint":null,"requested_detail":"location","response_style":"brief"}}

User: "What is in front of me?"
JSON: {{"intent":"describe_scene","target":null,"confidence":0.9,"language":"en","shelf_constraint":null,"requested_detail":"scene","response_style":"descriptive"}}
""".strip()

    def _to_intent_result(self, text: str, raw_response: str, parsed: dict[str, Any]) -> IntentResult:
        intent = coerce_intent(parsed.get("intent"))
        confidence = coerce_confidence(parsed.get("confidence"))
        if confidence < self.confidence_threshold:
            intent = Intent.UNKNOWN

        target = coerce_optional_string(parsed.get("target"))
        if target is not None:
            target = normalize_target(target)

        shelf_constraint = coerce_shelf_constraint(parsed.get("shelf_constraint"))

        return IntentResult(
            intent=intent,
            target=target,
            confidence=confidence,
            source="llm",
            language=coerce_optional_string(parsed.get("language")),
            shelf_constraint=shelf_constraint,
            requested_detail=coerce_optional_string(parsed.get("requested_detail")),
            response_style=coerce_optional_string(parsed.get("response_style")),
            raw_text=raw_response,
            metadata={"input_text": text},
        )


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    return value if isinstance(value, dict) else {}


def coerce_intent(value: Any) -> Intent:
    if not isinstance(value, str):
        return Intent.UNKNOWN

    try:
        return Intent(value.strip().lower())
    except ValueError:
        return Intent.UNKNOWN


def coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, confidence))


def coerce_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value or value.lower() in {"none", "null", "unknown", "n/a"}:
        return None

    return value


def coerce_shelf_constraint(value: Any) -> Optional[str]:
    value = coerce_optional_string(value)
    if value is None:
        return None

    normalized = value.lower().strip().replace(" ", "_").replace("-", "_")
    return normalized if normalized in ALLOWED_SHELF_CONSTRAINTS else value
