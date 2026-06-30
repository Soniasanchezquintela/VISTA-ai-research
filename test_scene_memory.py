"""
test_scene_memory.py

Tests TrackedShelfSceneMemory without any camera, YOLO, or CLIP.
Feeds fake detections frame by frame and checks that:
  - products persist across frames (keep same track ID)
  - new products get new track IDs
  - disappeared products are marked lost then removed
  - pointing (finger tip) correctly resolves to the right product

NOTE: This file is self-contained — it defines its own lightweight dataclasses
so it runs on Python 3.9 (the repo's types.py uses 3.10+ syntax).

Run:
    python test_scene_memory.py
"""

from __future__ import annotations

import pathlib
import sys
import types
from dataclasses import dataclass, field


# ── Minimal dataclasses (compatible with Python 3.9) ─────────────────────────

@dataclass
class ProductDetection:
    bbox: tuple
    confidence: float

@dataclass
class ProductIdentification:
    bbox: tuple
    sku_id: object
    score: float
    confidence: float
    description: object
    category: object
    accepted: bool

@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple
    last_seen_frame: int
    first_seen_frame: int
    missed_frames: int
    detection_score_ema: float
    sku_votes: dict
    best_sku_id: object
    best_sku_score: float
    identity_confidence: float
    description: object
    category: object
    price_eur: object
    state: str

@dataclass
class HandDetection:
    found: bool
    touched_point: tuple | None
    hand_landmarks: object = None


# ── Inject patched modules before importing tracked_scene_memory ──────────────

def _build_fake_cv2():
    """Minimal cv2 stub so we can import without opencv installed."""
    mod = types.ModuleType("cv2")
    mod.Mat = object
    mod.rectangle = lambda *a, **k: None
    mod.putText = lambda *a, **k: None
    mod.getTextSize = lambda *a, **k: ((10, 10), 2)
    mod.circle = lambda *a, **k: None
    mod.FONT_HERSHEY_SIMPLEX = 0
    mod.LINE_AA = 16
    return mod

# Patch sys.modules so our imports resolve cleanly
_fake_types_mod = types.ModuleType("scene_memory.types")
_fake_types_mod.ProductDetection = ProductDetection
_fake_types_mod.ProductIdentification = ProductIdentification
_fake_types_mod.TrackedObject = TrackedObject

_fake_sm_init = types.ModuleType("scene_memory")
_fake_sm_init.__path__ = [
    str(pathlib.Path(__file__).parent / "scene_memory")
]
_fake_sm_init.ProductDetection = ProductDetection
_fake_sm_init.ProductIdentification = ProductIdentification
_fake_sm_init.TrackedObject = TrackedObject

_fake_sm_core = types.ModuleType("scene_memory.scene_memory")
_fake_sm_core.annotate_box = lambda frame, box, label, color: frame  # no-op
_fake_hand_detector = types.ModuleType("hand_detector")
_fake_hand_detector.HandDetection = HandDetection
_fake_hand_detector.draw_hand_landmarks = lambda *a, **k: None
_fake_object_detector = types.ModuleType("object_detector")
_fake_object_detector.ProductDetection = ProductDetection

sys.modules.setdefault("cv2", _build_fake_cv2())
sys.modules["scene_memory"] = _fake_sm_init
sys.modules["scene_memory.types"] = _fake_types_mod
sys.modules["scene_memory.scene_memory"] = _fake_sm_core
sys.modules["hand_detector"] = _fake_hand_detector
sys.modules["object_detector"] = _fake_object_detector

import importlib.util

spec = importlib.util.spec_from_file_location(
    "scene_memory.tracked_scene_memory",
    pathlib.Path(__file__).parent / "scene_memory" / "tracked_scene_memory.py",
)
_tsm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_tsm_mod)

TrackedShelfSceneMemory = _tsm_mod.TrackedShelfSceneMemory


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_detection(bbox, confidence=0.9):
    return ProductDetection(bbox=bbox, confidence=confidence)

def make_identification(bbox, sku_id, description, score=0.85, confidence=0.9, accepted=True):
    return ProductIdentification(
        bbox=bbox, sku_id=sku_id, score=score, confidence=confidence,
        description=description, category="dairy-free", accepted=accepted,
    )

def make_hand_detection(finger_tip=None):
    return HandDetection(found=finger_tip is not None, touched_point=finger_tip)

def update_memory(mem, frame_index, detections, identifications, finger_tip=None):
    mem.update(frame_index, detections, identifications, make_hand_detection(finger_tip))

def separator(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)

def assert_eq(label, got, expected):
    status = "✅" if got == expected else "❌"
    print(f"  {status} {label}: got={got!r}, expected={expected!r}")
    assert got == expected, f"FAILED: {label}"

def assert_bbox_close(label, got, expected, precision=6):
    rounded_got = tuple(round(value, precision) for value in got)
    rounded_expected = tuple(round(value, precision) for value in expected)
    assert_eq(label, rounded_got, rounded_expected)


# ── Test 1: Products persist across frames ────────────────────────────────────
separator("Test 1 — Products persist across frames (same track ID)")

mem = TrackedShelfSceneMemory()
bbox_a = (100, 100, 200, 200)
bbox_b = (300, 100, 400, 200)

update_memory(mem, 1, [make_detection(bbox_a), make_detection(bbox_b)], [
    make_identification(bbox_a, "oatly_1l", "Oatly Oat Milk 1L"),
    make_identification(bbox_b, "alpro_soy", "Alpro Soy Milk"),
])
ids_frame1 = set(mem.tracks.keys())
print(f"  Frame 1 track IDs: {ids_frame1}")

bbox_a2 = (102, 101, 202, 201)  # slightly shifted — still overlaps well
bbox_b2 = (301, 100, 401, 200)

update_memory(mem, 2, [make_detection(bbox_a2), make_detection(bbox_b2)], [
    make_identification(bbox_a2, "oatly_1l", "Oatly Oat Milk 1L"),
    make_identification(bbox_b2, "alpro_soy", "Alpro Soy Milk"),
])
ids_frame2 = set(mem.tracks.keys())
print(f"  Frame 2 track IDs: {ids_frame2}")
assert_eq("Track IDs stable across frames", ids_frame1, ids_frame2)


# ── Test 2: New product gets a new track ID ───────────────────────────────────
separator("Test 2 — New product gets a new ID")

bbox_c = (500, 100, 600, 200)
update_memory(mem, 3, [make_detection(bbox_a2), make_detection(bbox_b2), make_detection(bbox_c)], [
    make_identification(bbox_a2, "oatly_1l", "Oatly Oat Milk 1L"),
    make_identification(bbox_b2, "alpro_soy", "Alpro Soy Milk"),
    make_identification(bbox_c, "yosoy_oat", "Yosoy Oat Milk"),
])
ids_frame3 = set(mem.tracks.keys())
new_ids = ids_frame3 - ids_frame2
print(f"  Frame 3 track IDs: {ids_frame3}, new: {new_ids}")
assert_eq("One new track created", len(new_ids), 1)


# ── Test 3: Disappeared product is removed after max_missed_frames ────────────
separator("Test 3 — Disappeared product removed after max_missed_frames")

for i in range(4, 4 + mem.max_missed_frames + 5):
    update_memory(mem, i, [make_detection(bbox_a2), make_detection(bbox_b2)], [
        make_identification(bbox_a2, "oatly_1l", "Oatly Oat Milk 1L"),
        make_identification(bbox_b2, "alpro_soy", "Alpro Soy Milk"),
    ])

new_id = list(new_ids)[0]
assert_eq(f"Track {new_id} (Yosoy) removed after disappearing", new_id in mem.tracks, False)


# ── Test 4: Finger tip inside a box → correct product ────────────────────────
separator("Test 4 — Finger tip inside bounding box")

mem2 = TrackedShelfSceneMemory()
bbox_x = (50, 50, 150, 150)
bbox_y = (200, 50, 300, 150)

update_memory(mem2, 1, [make_detection(bbox_x), make_detection(bbox_y)], [
    make_identification(bbox_x, "oatly_1l", "Oatly Oat Milk 1L"),
    make_identification(bbox_y, "alpro_soy", "Alpro Soy Milk"),
], finger_tip=(100, 100))  # inside bbox_x

touched = mem2.get_touched_object()
assert_eq("Touched object is Oatly", touched.description if touched else None, "Oatly Oat Milk 1L")


# ── Test 5: Finger tip near but outside → closest box ────────────────────────
separator("Test 5 — Finger tip near a box (within tolerance)")

update_memory(mem2, 2, [make_detection(bbox_x), make_detection(bbox_y)], [
    make_identification(bbox_x, "oatly_1l", "Oatly Oat Milk 1L"),
    make_identification(bbox_y, "alpro_soy", "Alpro Soy Milk"),
], finger_tip=(160, 100))  # 10px outside bbox_x (x2=150)

touched = mem2.get_touched_object()
assert_eq("Nearest box (Oatly) returned", touched.description if touched else None, "Oatly Oat Milk 1L")


# ── Test 6: Finger tip far away → None ───────────────────────────────────────
separator("Test 6 — Finger tip far away → no object touched")

update_memory(mem2, 3, [make_detection(bbox_x), make_detection(bbox_y)], [
    make_identification(bbox_x, "oatly_1l", "Oatly Oat Milk 1L"),
    make_identification(bbox_y, "alpro_soy", "Alpro Soy Milk"),
], finger_tip=(800, 800))

touched = mem2.get_touched_object()
assert_eq("No object touched when finger is far", touched, None)


# ── Test 7: find_by_description ──────────────────────────────────────────────
separator("Test 7 — find_by_description()")

# Use a fresh memory so both products are freshly visible (missed_frames=0)
mem_fd = TrackedShelfSceneMemory()
update_memory(mem_fd, 1, [make_detection(bbox_x), make_detection(bbox_y)], [
    make_identification(bbox_x, "oatly_1l", "Oatly Oat Milk 1L"),
    make_identification(bbox_y, "alpro_soy", "Alpro Soy Milk"),
])

result = mem_fd.find_by_description("alpro")
assert_eq("find_by_description('alpro') → Alpro", result.description if result else None, "Alpro Soy Milk")

result2 = mem_fd.find_by_description("coconut")
assert_eq("find_by_description('coconut') → None", result2, None)


# ── Test 8: tentative → confirmed after 3 frames ─────────────────────────────
separator("Test 8 — Track promotes tentative → confirmed after 3 frames")

mem3 = TrackedShelfSceneMemory()
bbox_z = (10, 10, 80, 80)
for i in range(1, 5):
    update_memory(mem3, i, [make_detection(bbox_z)],
                  [make_identification(bbox_z, "oatly_1l", "Oatly Oat Milk 1L")])

track = list(mem3.tracks.values())[0]
assert_eq("Track state is confirmed after 4 frames", track.state, "confirmed")


# ── Test 9: accepted identification confirms an existing unknown track ─────────
separator("Test 9 — Accepted match confirms existing unknown track")

mem4 = TrackedShelfSceneMemory()
bbox_unknown = (100.2, 100.3, 200.4, 200.5)
update_memory(mem4, 1, [make_detection(bbox_unknown)], [
    make_identification((100, 100, 200, 200), None, "Unknown product", accepted=False)
])

track = list(mem4.tracks.values())[0]
assert_eq("Track starts tentative", track.state, "tentative")
assert_eq("Track starts without best SKU", track.best_sku_id, None)

bbox_matched = (101.2, 100.8, 201.4, 200.9)
update_memory(mem4, 2, [make_detection(bbox_matched)], [
    make_identification((101, 101, 201, 201), "accepted_sku", "Accepted Product", accepted=True)
])

track = list(mem4.tracks.values())[0]
assert_eq("Accepted existing match is confirmed", track.state, "confirmed")
assert_eq("Accepted existing match sets best SKU", track.best_sku_id, "accepted_sku")
assert_bbox_close(
    "Matched bbox is smoothed 70/30",
    track.bbox,
    (
        0.7 * bbox_unknown[0] + 0.3 * bbox_matched[0],
        0.7 * bbox_unknown[1] + 0.3 * bbox_matched[1],
        0.7 * bbox_unknown[2] + 0.3 * bbox_matched[2],
        0.7 * bbox_unknown[3] + 0.3 * bbox_matched[3],
    ),
)

# ── Test 10: describe_scene delegates to the integrated describer ─────────────
separator("Test 10 — describe_scene uses the integrated scene describer")

empty_description = TrackedShelfSceneMemory().describe_scene()
assert_eq(
    "Empty scene description",
    empty_description,
    "Nada a la vista. ¿Está activada la cámara?",
)


class StubSceneDescriber:
    def __init__(self):
        self.tracks = None
        self.language = None
        self.pointed_track = None

    def describe(self, tracks, language="es"):
        self.tracks = list(tracks)
        self.language = language
        return "Descripción generada"

    def describe_pointed_product(self, track, language="es"):
        self.pointed_track = track
        self.language = language
        return "Producto señalado generado"


stub_describer = StubSceneDescriber()
mem2._scene_describer = stub_describer
assert_eq(
    "describe_scene delegates its return value",
    mem2.describe_scene(),
    "Descripción generada",
)
assert_eq(
    "describe_scene passes visible tracks",
    len(stub_describer.tracks),
    len(mem2.get_visible_objects()),
)
assert_eq("describe_scene requests Spanish", stub_describer.language, "es")

mem2.touched_track_id = 1
assert_eq(
    "describe_pointed_product delegates its return value",
    mem2.describe_pointed_product(),
    "Producto señalado generado",
)
assert_eq(
    "describe_pointed_product passes only the touched track",
    stub_describer.pointed_track.track_id,
    1,
)
assert_eq(
    "describe_pointed_product requests Spanish",
    stub_describer.language,
    "es",
)
assert_eq(
    "No pointed product uses the guidance fallback",
    TrackedShelfSceneMemory().describe_pointed_product(),
    (
        "Mano no detectada o el producto no está claro. "
        "Intenta apartar la mano brevemente y vuelve a intentarlo."
    ),
)


# ── Done ──────────────────────────────────────────────────────────────────────
separator("All tests passed ✅")
