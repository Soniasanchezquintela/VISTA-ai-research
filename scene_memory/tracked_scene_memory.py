"""
scene_memory/tracked_scene_memory.py

A drop-in replacement for ShelfSceneMemory that actually remembers products
across frames and correctly resolves which product the user is pointing at.

Usage (swap into project.py):
    from scene_memory.tracked_scene_memory import TrackedShelfSceneMemory
    scene_memory = TrackedShelfSceneMemory()

The public interface is identical to ShelfSceneMemory so nothing else needs
to change in project.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import ProductIdentification, TrackedObject
from .scene_memory import annotate_box
from hand_detector import draw_hand_landmarks, HandDetection
from object_detector import ProductDetection

# ── IoU helpers ───────────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _box_contains_point(bbox: tuple, point: tuple[int, int]) -> bool:
    """Return True if the point (px, py) falls inside the bounding box."""
    x1, y1, x2, y2 = bbox
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def _box_center(bbox: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _distance_point_to_box(point: tuple[int, int], bbox: tuple) -> float:
    """Euclidean distance from point to nearest edge of bbox (0 if inside)."""
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return float(np.sqrt(dx * dx + dy * dy))


def _smooth_bbox(previous_bbox: tuple, new_bbox: tuple, previous_weight: float) -> tuple:
    """Blend a matched bbox with its previous value to reduce annotation jitter."""
    new_weight = 1.0 - previous_weight
    return tuple(
        previous_weight * previous + new_weight * new
        for previous, new in zip(previous_bbox, new_bbox)
    )


def _find_identification_for_bbox(
    bbox: tuple,
    id_by_bbox: dict[tuple, ProductIdentification],
    identifications: list[ProductIdentification],
    iou_threshold: float,
) -> ProductIdentification | None:
    """Find the identification belonging to a detection bbox."""
    ident = id_by_bbox.get(bbox)
    if ident is not None:
        return ident

    best_ident = None
    best_iou = 0.0
    for candidate in identifications:
        iou = _iou(bbox, candidate.bbox)
        if iou > best_iou:
            best_iou = iou
            best_ident = candidate

    if best_iou >= iou_threshold:
        return best_ident
    return None


# ── Main class ────────────────────────────────────────────────────────────────

class TrackedShelfSceneMemory:
    """
    Persistent frame-to-frame scene memory for shelf product tracking.

    Key improvements over ShelfSceneMemory:
    - Products are matched across frames by bounding-box IoU — they keep
      their track ID and accumulated identity votes rather than being
      recreated from scratch every frame.
    - get_touched_object() actually checks which box the finger tip falls
      inside (or is closest to, as a fallback).
    - Products graduate from 'tentative' → 'confirmed' after being seen
      for enough consecutive frames, and are marked 'lost' when they
      disappear, only being removed after max_missed_frames.
    """

    # How many consecutive frames a track must be seen before it's confirmed
    CONFIRM_FRAMES = 3
    BBOX_PREVIOUS_WEIGHT = 0.7

    def __init__(
        self,
        max_missed_frames: int = 30,
        max_locked_missed_frames: int = 300,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_missed_frames = max_missed_frames
        self.max_locked_missed_frames = max_locked_missed_frames
        self.iou_threshold = iou_threshold

        self.tracks: dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self.touched_track_id: int | None = None
        self.touch_point: tuple[int, int] | None = None
        self.last_hand_detection: HandDetection | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
        self.touched_track_id = None
        self.touch_point = None
        self.last_hand_detection = None

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        frame_index: int,
        detections: list[ProductDetection],
        identifications: list[ProductIdentification],
        hand_detection: HandDetection,
        verbose: bool = True,
    ) -> None:
        
        self.touch_point = None if not hand_detection.found else hand_detection.touched_point
        self.last_hand_detection = hand_detection

        if verbose:
            if self.touch_point is not None:
                print(f"[SceneMemory.update] Frame {frame_index}: Finger tip at {self.touch_point}")
            # print identifications for debugging
            print(f"[SceneMemory.update] Frame {frame_index}: {len(identifications)} identifications")
            for idx, ident in enumerate(identifications):
                print(
                    f"[{idx}]  bbox={ident.bbox} sku_id={ident.sku_id} score={ident.score:.3f} accepted={ident.accepted}"
                )

        # Build a lookup from bbox → identification for this frame
        id_by_bbox: dict[tuple, ProductIdentification] = {
            ident.bbox: ident for ident in identifications
        }

        # ── Match incoming detections to existing tracks ───────────────────
        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()

        incoming_bboxes = [d.bbox for d in detections]

        for track_id, track in self.tracks.items():
            best_iou = 0.0
            best_det_idx = -1

            for det_idx, bbox in enumerate(incoming_bboxes):
                if det_idx in matched_detection_indices:
                    continue
                iou = _iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_iou >= self.iou_threshold and best_det_idx >= 0:
                # Matched — update the existing track
                matched_track_ids.add(track_id)
                matched_detection_indices.add(best_det_idx)

                new_bbox = incoming_bboxes[best_det_idx]
                ident = _find_identification_for_bbox(
                    new_bbox,
                    id_by_bbox,
                    identifications,
                    self.iou_threshold,
                )

                track.bbox = _smooth_bbox(
                    track.bbox,
                    new_bbox,
                    self.BBOX_PREVIOUS_WEIGHT,
                )
                track.last_seen_frame = frame_index
                track.missed_frames = 0

                # Revive a track that had been marked lost when it reappears
                if track.state == "lost":
                    track.state = "confirmed"

                # EMA on detection confidence
                det_conf = detections[best_det_idx].confidence
                track.detection_score_ema = (
                    0.7 * track.detection_score_ema + 0.3 * det_conf
                )

                # Accumulate identity votes
                if ident is not None and ident.accepted and ident.sku_id:
                    track.sku_votes[ident.sku_id] = (
                        track.sku_votes.get(ident.sku_id, 0.0) + ident.score
                    )
                    if verbose:
                        print("Assigning best SKU:", ident.sku_id, "with score:", track.sku_votes[ident.sku_id])
                    track.best_sku_id = ident.sku_id
                    track.best_sku_score = ident.score
                    track.identity_confidence = ident.confidence
                    track.description = ident.description
                    track.category = ident.category
                    track.state = "confirmed"

                # Promote to confirmed if seen enough times
                frames_alive = frame_index - track.first_seen_frame
                if track.state == "tentative" and frames_alive >= self.CONFIRM_FRAMES:
                    track.state = "confirmed"

        # ── Create new tracks for unmatched detections ─────────────────────
        new_track_ids: set[int] = set()
        for det_idx, detection in enumerate(detections):
            if det_idx in matched_detection_indices:
                continue

            ident = _find_identification_for_bbox(
                detection.bbox,
                id_by_bbox,
                identifications,
                self.iou_threshold,
            )
            accepted = ident is not None and ident.accepted and ident.sku_id is not None

            new_track = TrackedObject(
                track_id=self.next_track_id,
                bbox=detection.bbox,
                last_seen_frame=frame_index,
                first_seen_frame=frame_index,
                missed_frames=0,
                detection_score_ema=detection.confidence,
                sku_votes={ident.sku_id: ident.score} if accepted else {},
                best_sku_id=ident.sku_id if accepted else None,
                best_sku_score=ident.score if ident else 0.0,
                identity_confidence=ident.confidence if ident else 0.0,
                description=ident.description if ident else None,
                category=ident.category if ident else None,
                price_eur=None,
                state="tentative",
            )
            self.tracks[self.next_track_id] = new_track
            new_track_ids.add(self.next_track_id)
            self.next_track_id += 1

        # ── Age out unmatched tracks ───────────────────────────────────────
        to_remove = []
        for track_id, track in self.tracks.items():
            if track_id in matched_track_ids or track_id in new_track_ids:
                continue
            track.missed_frames += 1
            track.state = "lost"

            limit = (
                self.max_locked_missed_frames
                if track.state == "locked"
                else self.max_missed_frames
            )
            if track.missed_frames > limit:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

        # ── Resolve which track the finger is touching ─────────────────────
        self.touched_track_id = None
        if self.touch_point is not None:
            self._resolve_touched_track(self.touch_point)

        if verbose:
            print(
                f"[SceneMemory] frame={frame_index} | "
                f"tracks={len(self.tracks)} | "
                f"touched={self.touched_track_id} | "
                f"finger={self.touch_point}"
            )

    # ── Pointing resolution ───────────────────────────────────────────────────

    def _resolve_touched_track(self, finger_tip: tuple[int, int]) -> None:
        """
        Find the track the finger tip is pointing at.

        Priority:
        1. Finger tip is inside the bounding box (direct hit)
        2. Closest bounding box by distance (within 80px tolerance)
        """
        best_id = None
        best_dist = float("inf")

        for track_id, track in self.tracks.items():
            if _box_contains_point(track.bbox, finger_tip):
                self.touched_track_id = track_id
                return  # direct hit — done

            dist = _distance_point_to_box(finger_tip, track.bbox)
            if dist < best_dist:
                best_dist = dist
                best_id = track_id

        PROXIMITY_TOLERANCE_PX = 80
        if best_dist <= PROXIMITY_TOLERANCE_PX:
            self.touched_track_id = best_id

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_touched_object(self) -> TrackedObject | None:
        if self.touched_track_id is None:
            return None
        return self.tracks.get(self.touched_track_id)

    def get_visible_objects(self) -> list[TrackedObject]:
        """Return all tracks currently visible (tentative or confirmed, not lost)."""
        return [
            t for t in self.tracks.values()
            if t.state in ("confirmed", "tentative")
        ]

    def find_by_description(self, query: str) -> TrackedObject | None:
        """
        Find a visible track whose description contains the query string.
        Used by navigate_to_target and confirm_target_present.
        """
        query_lower = query.lower()
        for track in self.get_visible_objects():
            if track.description and query_lower in track.description.lower():
                return track
        return None

    @staticmethod
    def _is_unknown(track: TrackedObject) -> bool:
        """A track is 'unknown' if it has no accepted SKU identity."""
        return track.best_sku_id is None or track.best_sku_id == "unknown"

    def describe_scene(self) -> str:
        """
        Generate a structured Spanish scene description.

        Rules (matching the project's describe_scene contract):
        - Unknown products are not named; we count them and mention the total.
        - Identical known products are grouped and mentioned only once.
        """
        visible_objects = self.get_visible_objects()
        if len(visible_objects) == 0:
            return "Nada a la vista. ¿Está activada la cámara?"

        description = "La escena contiene "
        described_sku_ids: set = set()
        i = 0
        unknown_count = 0
        for track in visible_objects:
            if self._is_unknown(track):
                unknown_count += 1
                continue
            if track.best_sku_id in described_sku_ids:
                continue

            described_sku_ids.add(track.best_sku_id)
            if i > 0:
                description += ", "
            description += f"{track.description}"
            i += 1
        description += "."
        if unknown_count > 0:
            description += f" Además, hay {unknown_count} productos que no reconozco."
        return description

    def describe_pointed_product(self) -> str:
        """Generate a Spanish description of the product the user is pointing at."""
        touched_object = self.get_touched_object()
        if touched_object is None or self._is_unknown(touched_object):
            return (
                "Mano no detectada o el producto no está claro. "
                "Intenta apartar la mano brevemente y vuelve a intentarlo."
            )
        return f"El producto que está señalando es {touched_object.description}."

    # ── Annotation ────────────────────────────────────────────────────────────

    def annotate_image(self, frame, verbose: bool = True) -> cv2.Mat:
        annotated = frame.copy()
        if verbose:
            print(f"[SceneMemory] Annotating frame with {len(self.tracks)} tracked objects.")

        # Colors are BGR (OpenCV convention)
        for track in self.tracks.values():
            if verbose:
                if track.best_sku_id:
                    print(
                        f"[{track.track_id}] {track.description} ({track.category}) "
                        f"score={track.best_sku_score:.3f} conf={track.identity_confidence:.3f} "
                        f"state={track.state}"
                    )
                else:
                    print(f"[{track.track_id}] Unknown — state={track.state}")

            if track.track_id == self.touched_track_id:
                color = (255, 0, 0)   # blue — being pointed at
            elif track.state == "lost":
                color = (128, 128, 128)  # gray — remembered but currently occluded
            elif track.best_sku_id:
                color = (0, 255, 0)     # green — identified
            else:
                color = (0, 0, 255)     # red — unknown

            annotated = annotate_box(annotated, track.bbox, str(track.track_id), color)

        if self.touch_point is not None and self.last_hand_detection is not None:
            draw_hand_landmarks(annotated, self.last_hand_detection.hand_landmarks)
            cv2.circle(annotated, self.touch_point, radius=10, color=(255, 0, 0), thickness=-1)

        return annotated

    def forget_old_tracks(self) -> None:
        """Manually trigger removal of lost tracks (called externally if needed)."""
        to_remove = [
            tid for tid, t in self.tracks.items()
            if t.missed_frames > self.max_missed_frames
        ]
        for tid in to_remove:
            del self.tracks[tid]
