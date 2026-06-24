from .types import ProductDetection, ProductIdentification, TrackedObject

import cv2


def annotate_box(frame, box, label: str, color):
    annotated_frame = frame
    frame_height, frame_width = annotated_frame.shape[:2]
    line_thickness = max(3, min(frame_height, frame_width) // 200)
    font_scale = max(1.2, min(frame_height, frame_width) / 500)
    font_thickness = max(3, line_thickness)

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, line_thickness)

    (label_width, label_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )
    label_top = max(0, y1 - label_height - baseline - 16)
    label_bottom = label_top + label_height + baseline + 12
    label_right = x1 + label_width + 20

    cv2.rectangle(
        annotated_frame,
        (x1, label_top),
        (label_right, label_bottom),
        color,
        -1,
    )

    label_origin = (x1 + 10, label_bottom - baseline - 6)
    cv2.putText(
        annotated_frame,
        label,
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        font_thickness,
        cv2.LINE_AA,
    )

    return annotated_frame

class ShelfSceneMemory:
    def __init__(
        self,
        max_missed_frames: int = 30,
        max_locked_missed_frames: int = 300,
        iou_threshold: float = 0.3,
    ) -> None:
        self.tracks: dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self.touched_track_id: int | None = None
        self.touch_point: tuple[int, int] | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1
        self.touched_track_id = None
        self.touch_point = None

    def update(
        self,
        frame_index: int,
        detections: list[ProductDetection],
        identifications: list[ProductIdentification],
        finger_tip: tuple[int, int] | None = None,
    ) -> None:
        """
        Update scene memory using current frame results.
        """
        # For the moment, we only store the current frame's detections and identifications, without any tracking or memory.
        self.reset()

        self.touch_point = finger_tip

        # print how many detections and identifications we have in this frame
        print(f"[SceneMemory] Updating with {len(detections)} detections and {len(identifications)} identifications.")

        for product in identifications:
            track = TrackedObject(
                track_id=self.next_track_id,
                bbox=product.bbox,
                last_seen_frame=frame_index,
                first_seen_frame=frame_index,
                missed_frames=0,
                detection_score_ema=product.confidence,
                sku_votes={},
                best_sku_id=product.sku_id if product.accepted else "unknown",
                best_sku_score=product.score,
                identity_confidence=product.score,
                description=product.description,
                category=product.category,
                price_eur=None,  # Price is not provided in the current data
                state="tentative",  # Initial state
            )
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

    def annotate_image(self, frame, verbose: bool = True) -> cv2.Mat:
        """
        Annotate the image with bounding boxes and labels for tracked objects.
        """
        annotated_frame = frame.copy()
        # print how many tracks we have in memory
        print(f"[SceneMemory] Annotating frame with {len(self.tracks)} tracked objects.")


        for track in self.tracks.values():
            if verbose:
                if track.best_sku_id != "unknown":
                    print(f"[{track.track_id}] {track.description} ({track.category}), Score {track.best_sku_score:.4f}, Confidence {track.identity_confidence:.4f}")
                else:
                    print(f"[{track.track_id}] Unknown product, best: {track.description} ({track.category}), Score {track.best_sku_score:.4f}, Confidence {track.identity_confidence:.4f}")

            color = (0, 255, 0) if track.best_sku_id != "unknown" else (0, 0, 255)
            if track.track_id == self.touched_track_id:
                color = (255, 0, 0)
            annotated_frame = annotate_box(annotated_frame, track.bbox, str(track.track_id), color)

        # Draw a blue circle at the touch point for visualization
        if self.touch_point is not None:
            cv2.circle(annotated_frame, self.touch_point, radius=10, color=(255, 0, 0), thickness=-1)

        return annotated_frame

    def get_touched_object(self) -> TrackedObject | None:
        """
        Return the object the user is probably touching.
        """
        return None

    def get_visible_objects(self) -> list[TrackedObject]:
        """
        Return currently visible confirmed objects.
        """

        # For the moment, we return all objects in memory, just for testing
        return list(self.tracks.values())

    def describe_scene(self) -> str:
        """
        Generate a structured scene description.
        """
        visible_objects = self.get_visible_objects()
        if len(visible_objects) == 0:
            return "Nada a la vista. Está activada la cámara?"

        # NURIA: put here your describing scene code based on your LLM.
        # This is just a simple example that enumerates the detected products
        # with the following rules:
        # - if the product is unknown, we don't describe it, we count how many are there and say it.
        # - if we have several known products that are identical, we group them and only say it once.
        description = "La escena contiene "
        described_sku_ids: set[str] = set()
        i = 0
        unknown_count = 0
        for track in visible_objects:
            if track.best_sku_id == "unknown":
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
            description += " Además, hay " + f"{unknown_count} productos que no reconozco."
        return description

    def describe_pointed_product(self) -> str:
        """
        Generate a description of the product the user is pointing at.
        """
        touched_object = self.get_touched_object()
        if touched_object is None:
            return "Mano no detectada o el producto no está claro. Intenta a apartar la mano brevemente y vuelve a intentarlo."

        # NURIA: put here your describing pointed product code based on your LLM.
        # This is just a simple example that describes the detected product.
        return f"El producto que está señalando es {touched_object.description}."

    def forget_old_tracks(self) -> None:
        """
        Remove objects that have been missing for too long.
        """
