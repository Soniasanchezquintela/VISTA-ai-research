from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
