from __future__ import annotations

from collections import Counter
from typing import Iterable

from .types import TrackedObject


def is_unknown_track(track: TrackedObject) -> bool:
    return track.best_sku_id is None or track.best_sku_id == "unknown"


def track_center_y(track: TrackedObject) -> float:
    _x1, y1, _x2, y2 = track.bbox
    return (y1 + y2) / 2


def track_height(track: TrackedObject) -> float:
    _x1, y1, _x2, y2 = track.bbox
    return max(1.0, y2 - y1)


def estimate_shelf_threshold(tracks: list[TrackedObject]) -> float:
    if not tracks:
        return 0.0

    heights = sorted(track_height(track) for track in tracks)
    median_height = heights[len(heights) // 2]
    return max(40.0, median_height * 0.50)


def cluster_tracks_by_shelf(tracks: Iterable[TrackedObject]) -> list[dict]:
    sorted_tracks = sorted(tracks, key=track_center_y)
    if not sorted_tracks:
        return []

    threshold = estimate_shelf_threshold(sorted_tracks)
    clusters: list[dict] = []

    for track in sorted_tracks:
        center_y = track_center_y(track)
        if not clusters:
            clusters.append({"tracks": [track], "center_y": center_y})
            continue

        nearest = min(
            clusters,
            key=lambda cluster: abs(center_y - cluster["center_y"]),
        )
        if abs(center_y - nearest["center_y"]) <= threshold:
            nearest["tracks"].append(track)
            nearest["center_y"] = sum(
                track_center_y(item) for item in nearest["tracks"]
            ) / len(nearest["tracks"])
        else:
            clusters.append({"tracks": [track], "center_y": center_y})

    clusters.sort(key=lambda cluster: cluster["center_y"])
    shelf_count = len(clusters)

    for index, cluster in enumerate(clusters, start=1):
        cluster["shelf_index"] = index
        cluster["label_es"] = shelf_label_es(index, shelf_count)
        cluster["label_en"] = shelf_label_en(index, shelf_count)
        cluster["tracks"].sort(
            key=lambda track: (track.bbox[0] + track.bbox[2]) / 2
        )

    return clusters


def shelf_label_es(shelf_index: int, shelf_count: int) -> str:
    if shelf_count == 1:
        return "la única balda detectada"
    if shelf_index == 1:
        return "la balda superior"
    if shelf_index == shelf_count:
        return "la balda inferior"
    return f"la balda {shelf_index} desde arriba"


def shelf_label_en(shelf_index: int, shelf_count: int) -> str:
    if shelf_count == 1:
        return "the only detected shelf"
    if shelf_index == 1:
        return "top shelf"
    if shelf_index == shelf_count:
        return "bottom shelf"
    return f"shelf {shelf_index} from the top"


def counted_names_text(
    names: list[str],
    possible: bool,
    language: str,
) -> list[str]:
    parts = []
    for name, count in Counter(names).most_common():
        if possible:
            label = f"posible {name}" if language == "es" else f"possible {name}"
        else:
            label = name

        if count == 1:
            parts.append(label)
        else:
            parts.append(f"{count} {label}")

    return parts


def join_parts(parts: list[str], language: str) -> str:
    if not parts:
        return (
            "ningún producto identificado"
            if language == "es"
            else "no identified products"
        )
    if len(parts) == 1:
        return parts[0]

    conjunction = " y " if language == "es" else " and "
    return ", ".join(parts[:-1]) + conjunction + parts[-1]


def summarize_tracks(
    tracks: list[TrackedObject],
    language: str = "es",
) -> str:
    known_names = []
    possible_names = []
    unknown_count = 0

    for track in tracks:
        description = track.description
        if is_unknown_track(track):
            if description:
                possible_names.append(description)
            else:
                unknown_count += 1
            continue

        if description:
            known_names.append(description)
        else:
            unknown_count += 1

    parts = []
    parts.extend(
        counted_names_text(known_names, possible=False, language=language)
    )
    parts.extend(
        counted_names_text(possible_names, possible=True, language=language)
    )

    if unknown_count:
        if language == "es":
            label = (
                "1 producto que no reconozco"
                if unknown_count == 1
                else f"{unknown_count} productos que no reconozco"
            )
        else:
            label = (
                "1 unidentified product"
                if unknown_count == 1
                else f"{unknown_count} unidentified products"
            )
        parts.append(label)

    return join_parts(parts, language)


def describe_pointed_track(
    track: TrackedObject,
    language: str = "es",
) -> str:
    """Return only grounded, user-facing facts about one selected product."""
    if language == "es":
        name = track.description or "un producto identificado"
        facts = [f"Producto señalado: {name}"]
        if track.category:
            facts.append(f"categoría: {track.category}")
        if track.price_eur is not None:
            price = f"{track.price_eur:.2f}".replace(".", ",")
            facts.append(f"precio: {price} euros")
    else:
        name = track.description or "an identified product"
        facts = [f"Pointed product: {name}"]
        if track.category:
            facts.append(f"category: {track.category}")
        if track.price_eur is not None:
            facts.append(f"price: {track.price_eur:.2f} euros")

    return "; ".join(facts) + "."


def describe_tracks_by_shelf(
    tracks: Iterable[TrackedObject],
    language: str = "es",
) -> str:
    visible_tracks = list(tracks)
    if not visible_tracks:
        return (
            "Nada a la vista. ¿Está activada la cámara?"
            if language == "es"
            else "I do not see any products clearly."
        )

    clusters = cluster_tracks_by_shelf(visible_tracks)

    if language == "es":
        shelf_parts = [
            (
                f"{cluster['label_es']}: "
                f"{summarize_tracks(cluster['tracks'], language='es')}"
            )
            for cluster in clusters
        ]
        shelf_count_text = (
            "1 balda" if len(clusters) == 1 else f"{len(clusters)} baldas"
        )
        return f"Detecto {shelf_count_text}. " + "; ".join(shelf_parts) + "."

    shelf_parts = [
        (
            f"{cluster['label_en']}: "
            f"{summarize_tracks(cluster['tracks'], language='en')}"
        )
        for cluster in clusters
    ]
    return f"I detected {len(clusters)} shelves. " + "; ".join(shelf_parts) + "."
