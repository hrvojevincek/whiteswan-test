"""
Interactive helper to build the reference gallery from input video.

Usage:
  python build_gallery.py --input input.mp4 --gallery gallery

Controls (OpenCV window must be focused):
  n / RIGHT  - next frame (+1)
  p / LEFT   - previous frame (-1)
  j          - jump forward 30 frames
  k          - jump back 30 frames
  d          - detect faces on current frame (press d; default detector is fast opencv)

  1-9        - select face # (yellow box); use with 2+ faces, or confirms #1 when alone
  [ / ]      - previous / next selected face
  h r e m s  - save face to Harry / Ron / Hermione / McGonagall / Snape (only these save)

  q / ESC    - quit
"""

import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected

from annotate_video import CHARACTERS, _safe_xyxy

# Reject DeepFace's "no face" fallback (whole frame, confidence 0) and other bogus boxes.
_MAX_FACE_AREA_RATIO = 0.45
_MIN_FACE_SIDE_PX = 40


KEY_TO_SLUG: Dict[str, str] = {
    "h": "harry",
    "r": "ron",
    "e": "hermione",
    "m": "mcgonagall",
    "s": "snape",
}


def _is_plausible_face(
    face_obj: dict,
    frame_w: int,
    frame_h: int,
) -> bool:
    """Drop full-frame fallbacks and other invalid detections."""
    confidence = float(face_obj.get("confidence") or 0)
    if confidence <= 0:
        return False

    area = face_obj.get("facial_area", {})
    bbox = _safe_xyxy(
        int(area.get("x", 0)),
        int(area.get("y", 0)),
        int(area.get("w", 0)),
        int(area.get("h", 0)),
        frame_w,
        frame_h,
    )
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < _MIN_FACE_SIDE_PX or bh < _MIN_FACE_SIDE_PX:
        return False

    frame_area = frame_w * frame_h
    if frame_area <= 0:
        return False
    if (bw * bh) / frame_area > _MAX_FACE_AREA_RATIO:
        return False
    if bw > 0.92 * frame_w and bh > 0.92 * frame_h:
        return False
    return True


def _detect_faces(
    frame_bgr: np.ndarray,
    detector_backend: str = "opencv",
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    h, w = frame_bgr.shape[:2]
    try:
        face_objs = DeepFace.extract_faces(
            img_path=frame_bgr,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True,
            color_face="bgr",
            normalize_face=False,  # keep uint8 0-255 for cv2.imwrite (True → black JPGs)
        )
    except FaceNotDetected:
        return []

    results: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
    for face_obj in face_objs:
        if not _is_plausible_face(face_obj, w, h):
            continue
        area = face_obj.get("facial_area", {})
        bbox = _safe_xyxy(
            int(area.get("x", 0)),
            int(area.get("y", 0)),
            int(area.get("w", 0)),
            int(area.get("h", 0)),
            w,
            h,
        )
        face_img = face_obj.get("face")
        if face_img is not None:
            results.append((bbox, face_img))
    return results


def _put_ui_line(vis: np.ndarray, text: str, y: int, scale: float = 0.5, bold: bool = False) -> int:
    """Draw one ASCII-only UI line (OpenCV font cannot render unicode dashes)."""
    thickness = 2 if bold else 1
    cv2.putText(
        vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return y + int(22 * scale + 8)


def _help_lines_for_state(
    detections: List,
    selected_face_idx: int,
    frame_idx: int,
    total_frames: int,
    detector_backend: str,
) -> List[str]:
    lines = [f"FRAME {frame_idx} / {total_frames}"]

    if not detections:
        lines.extend(
            [
                "STEP 1: Press D  ->  find faces on this frame",
                "STEP 2: Press N  ->  go to next frame (anytime)",
                "Also: J/K = skip 30 frames   P = previous   Q = quit",
            ]
        )
        return lines

    n = len(detections)
    if n == 1:
        lines.append("1 face found (green box)")
        lines.extend(
            [
                "STEP 2: Press H R E M or S  ->  save to character",
                "         H=Harry  R=Ron  E=Hermione  M=McGonagall  S=Snape",
                "STEP 3: Press N  ->  next frame (continue video)",
            ]
        )
        lines.append(
            "Missing faces? Use a frame where all faces look at camera,"
            " or re-run with --detector retinaface (default)."
            if detector_backend == "opencv"
            else "Missing faces? Skip (J/N) to a frame with clear frontal faces."
        )
    else:
        sel = selected_face_idx + 1
        lines.append(f"{n} faces: yellow = selected (#{sel}), green = others")
        lines.extend(
            [
                "STEP 2: Press 1 2 3 ...  ->  pick face (yellow box)",
                "         Press [ or ]   ->  cycle selected face",
                "STEP 3: Press H R E M or S  ->  save selected face",
                "         H=Harry  R=Ron  E=Hermione  M=McGonagall  S=Snape",
                "STEP 4: Press N  ->  next frame (continue video)",
            ]
        )
    lines.append("J/K = skip 30 frames   D = re-detect   Q = quit")
    return lines


def _draw_preview(
    frame_bgr: np.ndarray,
    detections: List[Tuple[Tuple[int, int, int, int], np.ndarray]],
    frame_idx: int,
    total_frames: int,
    selected_face_idx: int = 0,
    detector_backend: str = "retinaface",
) -> np.ndarray:
    vis = frame_bgr.copy()
    for i, (bbox, _) in enumerate(detections):
        x1, y1, x2, y2 = bbox
        is_selected = len(detections) > 1 and i == selected_face_idx
        color = (0, 255, 255) if is_selected else (0, 255, 0)
        thickness = 3 if is_selected else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color=color, thickness=thickness)
        label = f"#{i + 1}"
        if is_selected:
            label += " SEL"
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    help_lines = _help_lines_for_state(
        detections, selected_face_idx, frame_idx, total_frames, detector_backend
    )
    panel_h = min(vis.shape[0], 28 + len(help_lines) * 26)
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (vis.shape[1], panel_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)

    y = 22
    for i, line in enumerate(help_lines):
        y = _put_ui_line(vis, line, y, scale=0.52 if i == 0 else 0.48, bold=(i == 0))
    return vis


def _next_filename(folder: str, slug: str) -> str:
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.startswith(slug) and f.lower().endswith((".jpg", ".png"))]
    n = len(existing) + 1
    return os.path.join(folder, f"{slug}_{n:02d}.jpg")


def _save_face(gallery_dir: str, slug: str, face_bgr: np.ndarray) -> str:
    out_dir = os.path.join(gallery_dir, slug)
    path = _next_filename(out_dir, slug)
    face = np.asarray(face_bgr)
    if face.dtype != np.uint8 or float(face.max()) <= 1.0:
        face = (np.clip(face, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, face)
    return path


def run(
    input_video: str,
    gallery_dir: str,
    start_frame: int = 0,
    detector_backend: str = "opencv",
) -> None:
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(start_frame, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    detections: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
    detection_cache: Dict[int, List[Tuple[Tuple[int, int, int, int], np.ndarray]]] = {}
    selected_face_idx = 0
    window = "build_gallery"

    print("[INFO] Click the 'build_gallery' window, then:")
    print("       D = detect faces   H/R/E/M/S = save to character   N = next frame")
    print(f"       Detector: {detector_backend}")
    if detector_backend == "opencv":
        print("[WARN] opencv often misses side/profile faces. Prefer: --detector retinaface")
    elif detector_backend == "retinaface":
        print("[INFO] RetinaFace is slow on CPU; wait after D before pressing other keys.")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] Cannot read frame {frame_idx}")
            break

        # Restore cached detections for this frame, or show raw frame until user presses d.
        detections = detection_cache.get(frame_idx, [])
        if selected_face_idx >= len(detections):
            selected_face_idx = max(0, len(detections) - 1)

        preview = _draw_preview(
            frame, detections, frame_idx, total, selected_face_idx, detector_backend
        )
        cv2.imshow(window, preview)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            break
        if key in (ord("n"), 83, 3):  # n or right arrow
            frame_idx = min(frame_idx + 1, total - 1)
            selected_face_idx = 0
            continue
        if key in (ord("p"), 81, 2):  # p or left arrow
            frame_idx = max(frame_idx - 1, 0)
            selected_face_idx = 0
            continue
        if key == ord("j"):
            frame_idx = min(frame_idx + 30, total - 1)
            selected_face_idx = 0
            continue
        if key == ord("k"):
            frame_idx = max(frame_idx - 30, 0)
            selected_face_idx = 0
            continue
        if key == ord("]") and detections:
            selected_face_idx = (selected_face_idx + 1) % len(detections)
            continue
        if key == ord("[") and detections:
            selected_face_idx = (selected_face_idx - 1) % len(detections)
            continue
        if key == ord("d"):
            print(f"[INFO] Detecting faces on frame {frame_idx} ({detector_backend})...")
            detections = _detect_faces(frame, detector_backend=detector_backend)
            detection_cache[frame_idx] = detections
            if detections:
                print(f"[INFO] Found {len(detections)} face(s) on frame {frame_idx}.")
            else:
                print(f"[INFO] No faces on frame {frame_idx} — press N/J to try another frame.")
            continue

        ch = chr(key) if 32 <= key < 127 else ""

        # Pick which face (1-9). Never saves — only H/R/E/M/S save to a character.
        if ch.isdigit() and detections:
            digit = int(ch)
            if digit == 0:
                continue
            if 1 <= digit <= len(detections):
                selected_face_idx = digit - 1
                print(f"[INFO] Selected face #{digit} (then press H/R/E/M/S to save)")
            else:
                print(f"[WARN] Face #{digit} not found ({len(detections)} face(s) on this frame)")
            continue

        if ch in KEY_TO_SLUG:
            slug = KEY_TO_SLUG[ch]
            if not detections:
                print("[WARN] No faces detected. Press 'd' to detect first.")
                continue
            face_idx = 0 if len(detections) == 1 else selected_face_idx
            _, face_img = detections[face_idx]
            path = _save_face(gallery_dir, slug, face_img)
            print(
                f"[INFO] Saved {CHARACTERS[slug]} (face #{face_idx + 1}) -> {path}"
            )

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done. Re-run annotate_video.py when gallery has enough crops per character.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reference gallery from video frames.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--gallery", default="gallery", help="Gallery root directory")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to start on")
    parser.add_argument(
        "--detector",
        default="retinaface",
        choices=["opencv", "retinaface", "ssd", "mtcnn"],
        help="Face detector (default retinaface; matches annotate_video). Use opencv only if speed matters.",
    )
    args = parser.parse_args()
    run(
        args.input,
        args.gallery,
        start_frame=args.start_frame,
        detector_backend=args.detector,
    )


if __name__ == "__main__":
    main()
