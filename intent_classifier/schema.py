from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class Intent(str, Enum):
    DESCRIBE_SCENE = "describe_scene"
    DESCRIBE_POINTED_PRODUCT = "describe_pointed_product"
    NAVIGATE_TO_TARGET = "navigate_to_target"
    CONFIRM_TARGET_PRESENT = "confirm_target_present"
    GET_PRICE = "get_price"
    READ_TEXT = "read_text"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    intent: Intent
    target: Optional[str]
    confidence: float
    source: str = "bert"
    language: Optional[str] = None
    shelf_constraint: Optional[str] = None
    requested_detail: Optional[str] = None
    response_style: Optional[str] = None
    raw_text: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
