
from dataclasses import dataclass


@dataclass
class ProductDetection:
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class ProductIdentification:
    bbox: tuple[float, float, float, float]
    sku_id: str | None
    score: float
    confidence: float
    description: str | None
    category: str | None
    accepted: bool


@dataclass
class TrackedObject:
    track_id: int

    # Geometry
    bbox: tuple[float, float, float, float]
    last_seen_frame: int
    first_seen_frame: int
    missed_frames: int

    # Detection confidence
    detection_score_ema: float

    # Identity memory
    sku_votes: dict[str, float]
    best_sku_id: str | None
    best_sku_score: float
    identity_confidence: float

    # Optional metadata
    description: str | None
    category: str | None
    price_eur: float | None

    # State
    state: str  # tentative, confirmed, lost, locked
