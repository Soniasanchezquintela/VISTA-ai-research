from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch import nn

from intent_classifier.rule_extractor import extract_target
from intent_classifier.schema import Intent, IntentResult


MODEL_NAME = "distilbert-base-multilingual-cased"

INTENTS: Sequence[Intent] = (
    Intent.DESCRIBE_SCENE,
    Intent.DESCRIBE_POINTED_PRODUCT,
    Intent.NAVIGATE_TO_TARGET,
    Intent.CONFIRM_TARGET_PRESENT,
    Intent.GET_PRICE,
    Intent.READ_TEXT,
    Intent.UNKNOWN,
)

INTENT_TO_ID: Dict[Intent, int] = {intent: index for index, intent in enumerate(INTENTS)}
ID_TO_INTENT: Dict[int, Intent] = {index: intent for intent, index in INTENT_TO_ID.items()}


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class IntentClassifierModel(nn.Module):
    """DistilBERT encoder followed by an MLP intent classification head."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = len(INTENTS),
        hidden_size: int = 256,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "IntentClassifierModel requires the 'transformers' package. "
                "Install the project requirements before creating the model."
            ) from exc

        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = encoder_output.last_hidden_state[:, 0]
        return self.classifier(cls_embedding)


class IntentClassifier:
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_name: str = MODEL_NAME,
        confidence_threshold: float = 0.5,
        max_length: int = 64,
        hidden_size: int = 256,
        dropout: float = 0.2,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        checkpoint = None
        state_dict = None
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                model_name = checkpoint.get("model_name", model_name)
                max_length = checkpoint.get("max_length", max_length)
                hidden_size = checkpoint.get("hidden_size", hidden_size)
                dropout = checkpoint.get("dropout", dropout)
            else:
                state_dict = checkpoint

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "IntentClassifier requires the 'transformers' package. "
                "Install the project requirements before running predictions."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device(device) if device is not None else get_default_device()
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length

        self.model = IntentClassifierModel(
            model_name=model_name,
            hidden_size=hidden_size,
            dropout=dropout,
        ).to(self.device)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, text: str) -> IntentResult:
        if not text.strip():
            return IntentResult(intent=Intent.UNKNOWN, target=None, confidence=0.0)

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        with torch.no_grad():
            logits = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        confidence, label_id = torch.max(probabilities, dim=-1)
        confidence_value = float(confidence.item())
        intent = ID_TO_INTENT[int(label_id.item())]

        if confidence_value < self.confidence_threshold:
            intent = Intent.UNKNOWN

        target = None
        if intent in {Intent.NAVIGATE_TO_TARGET, Intent.CONFIRM_TARGET_PRESENT}:
            target = extract_target(text)

        result = IntentResult(
            intent=intent,
            target=target,
            confidence=confidence_value,
        )
        return result

    def predict_batch(self, texts: Iterable[str]) -> List[IntentResult]:
        return [self.predict(text) for text in texts]


class HybridIntentClassifier:
    """Use BERT for fast known intents, then LLM for richer fallback parsing."""

    TARGET_INTENTS = {
        Intent.NAVIGATE_TO_TARGET,
        Intent.CONFIRM_TARGET_PRESENT,
        Intent.GET_PRICE,
    }

    def __init__(
        self,
        bert_classifier: IntentClassifier,
        llm_parser,
        llm_fallback_threshold: float = 0.70,
        prefer_llm: bool = False,
    ) -> None:
        self.bert_classifier = bert_classifier
        self.llm_parser = llm_parser
        self.llm_fallback_threshold = llm_fallback_threshold
        self.prefer_llm = prefer_llm

    def predict(self, text: str) -> IntentResult:
        if self.prefer_llm:
            return self.llm_parser.parse(text)

        bert_result = self.bert_classifier.predict(text)
        if not self._should_use_llm(text, bert_result):
            return bert_result

        llm_result = self.llm_parser.parse(text)
        return self._choose_result(bert_result, llm_result)

    def predict_batch(self, texts: Iterable[str]) -> List[IntentResult]:
        return [self.predict(text) for text in texts]

    def _should_use_llm(self, text: str, bert_result: IntentResult) -> bool:
        if not text.strip():
            return False

        if bert_result.intent == Intent.UNKNOWN:
            return True

        if bert_result.confidence < self.llm_fallback_threshold:
            return True

        if bert_result.intent in self.TARGET_INTENTS and not bert_result.target:
            return True

        return False

    def _choose_result(self, bert_result: IntentResult, llm_result: IntentResult) -> IntentResult:
        if llm_result.intent == Intent.UNKNOWN and bert_result.intent != Intent.UNKNOWN:
            return bert_result

        if llm_result.intent == bert_result.intent:
            if not llm_result.target and bert_result.target:
                llm_result.target = bert_result.target
            return llm_result

        if bert_result.confidence >= self.llm_fallback_threshold and llm_result.confidence < 0.80:
            return bert_result

        return llm_result
