from __future__ import annotations

from typing import Iterable

from .descriptions import describe_pointed_track, describe_tracks_by_shelf
from .types import TrackedObject


DEFAULT_SCENE_MODEL = "gemma3:4b"


class LLMSceneDescriber:
    """Turn grounded scene-memory facts into a natural spoken description."""

    def __init__(
        self,
        model: str = DEFAULT_SCENE_MODEL,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature

    def describe(
        self,
        tracks: Iterable[TrackedObject],
        language: str = "es",
    ) -> str:
        visible_tracks = list(tracks)
        grounded_description = describe_tracks_by_shelf(
            visible_tracks,
            language=language,
        )

        # There is nothing for the LLM to improve, and skipping it avoids a
        # needless model call when the camera has not produced detections.
        if not visible_tracks:
            return grounded_description

        return self._rewrite(
            grounded_description,
            self._scene_system_prompt(language),
        )

    def describe_pointed_product(
        self,
        track: TrackedObject,
        language: str = "es",
    ) -> str:
        """Describe only the product selected by the pointing detector."""
        grounded_description = describe_pointed_track(track, language=language)
        return self._rewrite(
            grounded_description,
            self._pointed_product_system_prompt(language),
        )

    def _rewrite(self, grounded_description: str, system_prompt: str) -> str:
        try:
            from ollama import chat

            response = chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (
                            "These are the only trusted scene facts:\n"
                            f"{grounded_description}"
                        ),
                    },
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": 120,
                },
            )
            content = self._response_content(response).strip()
        except Exception:
            return grounded_description

        return content or grounded_description

    @staticmethod
    def _scene_system_prompt(language: str) -> str:
        output_language = "Spanish" if language == "es" else "English"
        return f"""
You describe a supermarket shelf aloud for a blind user.

Rewrite the supplied structured scene facts as a short, natural description in
{output_language}. Preserve shelf order, product names, quantities, and all
uncertainty words. Mention every supplied item exactly once. Do not invent or
infer products, rows, brands, locations, properties, or advice. In particular,
do not turn "the only detected shelf" into "the first shelf." Use one or two
concise sentences. Return only the description, without headings or commentary.
""".strip()

    @staticmethod
    def _pointed_product_system_prompt(language: str) -> str:
        output_language = "Spanish" if language == "es" else "English"
        return f"""
You describe the single supermarket product a blind user is pointing at.

Rewrite the supplied trusted product facts as one short, natural sentence in
{output_language}. Describe only that product and preserve its name, category,
price, and all uncertainty words exactly when supplied. Do not invent or infer
its appearance, packaging, ingredients, nutrition, properties, location,
suitability, or advice. Return only the description, without headings or
commentary.
""".strip()

    @staticmethod
    def _response_content(response) -> str:
        if isinstance(response, dict):
            return response["message"]["content"]

        message = response.message
        if isinstance(message, dict):
            return message["content"]
        return message.content
