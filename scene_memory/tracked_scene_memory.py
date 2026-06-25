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

from .types import ProductDetection, ProductIdentification, TrackedObject
from .scene_memory import annotate_box


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

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
        self.touched_track_id = None
        self.touch_point = None

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        frame_index: int,
        detections: list[ProductDetection],
        identifications: list[ProductIdentification],
        finger_tip: tuple[int, int] | None = None,
    ) -> None:
        self.touch_point = finger_tip

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
                ident = id_by_bbox.get(new_bbox)

                track.bbox = new_bbox
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
                    # Pick the SKU with the highest cumulative vote
                    best_sku = max(track.sku_votes, key=track.sku_votes.get)
                    track.best_sku_id = best_sku
                    track.best_sku_score = ident.score
                    track.identity_confidence = ident.confidence
                    track.description = ident.description
                    track.category = ident.category

                # Promote to confirmed if seen enough times
                frames_alive = frame_index - track.first_seen_frame
                if track.state == "tentative" and frames_alive >= self.CONFIRM_FRAMES:
                    track.state = "confirmed"

        # ── Create new tracks for unmatched detections ─────────────────────
        new_track_ids: set[int] = set()
        for det_idx, detection in enumerate(detections):
            if det_idx in matched_detection_indices:
                continue

            ident = id_by_bbox.get(detection.bbox)
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
        if finger_tip is not None:
            self._resolve_touched_track(finger_tip)

        print(
            f"[SceneMemory] frame={frame_index} | "
            f"tracks={len(self.tracks)} | "
            f"touched={self.touched_track_id} | "
            f"finger={finger_tip}"
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
                color = (0, 165, 255)   # orange — being pointed at
            elif track.state == "lost":
                color = (128, 128, 128)  # gray — remembered but currently occluded
            elif track.best_sku_id:
                color = (0, 255, 0)     # green — identified
            else:
                color = (0, 0, 255)     # red — unknown

            annotated = annotate_box(annotated, track.bbox, str(track.track_id), color)

        if self.touch_point is not None:
            cv2.circle(annotated, self.touch_point, radius=10, color=(0, 165, 255), thickness=-1)

        return annotated

    def forget_old_tracks(self) -> None:
        """Manually trigger removal of lost tracks (called externally if needed)."""
        to_remove = [
            tid for tid, t in self.tracks.items()
            if t.missed_frames > self.max_missed_frames
        ]
        for tid in to_remove:
            del self.tracks[tid]
