from __future__ import annotations

from typing import Optional


def parse_face_bbox(value: str) -> Optional[tuple[int, int, int, int]]:
    value = (value or "").strip()
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("WANMUSE_FACE_BBOX must be x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("WANMUSE_FACE_BBOX values must be integers") from exc
    if x2 <= x1 or y2 <= y1:
        raise ValueError("WANMUSE_FACE_BBOX must have x2>x1 and y2>y1")
    return x1, y1, x2, y2
