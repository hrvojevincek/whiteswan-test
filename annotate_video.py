import argparse
import os
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm


UNKNOWN_LABEL = "Unknown"

# Slugs are the folder names under `--gallery`.
# Display names are what we draw on the video.
CHARACTERS: Dict[str, str] = {
    "harry": "Harry Potter",
    "ron": "Ron Weasley",
    "hermione": "Hermione Granger",
    "mcgonagall": "Prof. McGonagall",
    "snape": "Prof. Severus Snape",
}


@dataclass
class Track:
    track_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    last_seen_frame_idx: int
    labels_window: Deque[str]


def _iter_image_files(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files: List[str] = []
    for name in sorted(os.listdir(folder)):
        _, ext = os.path.splitext(name)
        if ext.lower() in exts:
            files.append(os.path.join(folder, name))
    return files


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # If DeepFace already L2-normalizes, the denominator is ~1.
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0
    cos_sim = float(np.dot(a, b) / denom)
    return 1.0 - cos_sim


def _majority_vote(labels: Deque[str]) -> str:
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = {lab for lab, c in counts.items() if c == max_count}

    # Tie-break: pick the most recent label among the tied candidates.
    for lab in reversed(labels):
        if lab in candidates:
            return lab
    return next(iter(candidates))


def _bbox_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 1e-12:
        return 0.0
    return float(inter_area / union)


def _safe_xyxy(
    x: int, y: int, w: int, h: int, frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w - 1, x + w)
    y2 = min(frame_h - 1, y + h)
    return (x1, y1, x2, y2)


def _is_plausible_detection(face_obj: dict, frame_w: int, frame_h: int) -> bool:
    """Skip DeepFace's no-face fallback (full frame, confidence 0)."""
    if float(face_obj.get("confidence") or 0) <= 0:
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
    if bw < 40 or bh < 40:
        return False
    frame_area = frame_w * frame_h
    if frame_area <= 0:
        return False
    if (bw * bh) / frame_area > 0.45:
        return False
    return True


def _extract_embedding_facenet512(face_img_bgr: np.ndarray) -> np.ndarray:
    # Crop is already from RetinaFace — skip re-detection (much faster on CPU).
    reps = DeepFace.represent(
        img_path=face_img_bgr,
        model_name="Facenet512",
        detector_backend="skip",
        enforce_detection=False,
        align=False,
        l2_normalize=True,
    )

    if isinstance(reps, dict):
        reps = [reps]
    if not reps:
        raise RuntimeError("DeepFace.represent returned no embeddings.")

    emb = np.asarray(reps[0]["embedding"], dtype=np.float32)
    return emb


def _precompute_gallery_embeddings(
    gallery_dir: str,
) -> Dict[str, List[np.ndarray]]:
    gallery_embeddings: Dict[str, List[np.ndarray]] = {slug: [] for slug in CHARACTERS}

    for slug, display_name in CHARACTERS.items():
        char_dir = os.path.join(gallery_dir, slug)
        if not os.path.isdir(char_dir):
            print(f"[WARN] Missing gallery folder: {char_dir} (display={display_name})")
            continue

        img_paths = _iter_image_files(char_dir)
        if not img_paths:
            print(f"[WARN] No images found in gallery folder: {char_dir}")
            continue

        print(f"[INFO] Embedding gallery: {slug} ({display_name}) — {len(img_paths)} images")
        for img_path in img_paths:
            reps = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet512",
                detector_backend="skip",
                enforce_detection=False,
                align=False,
                l2_normalize=True,
            )

            if isinstance(reps, dict):
                reps = [reps]
            if not reps:
                continue

            emb = np.asarray(reps[0]["embedding"], dtype=np.float32)
            gallery_embeddings[slug].append(emb)

    return gallery_embeddings


def _best_of_character_match(
    query_embedding: np.ndarray,
    gallery_embeddings: Dict[str, List[np.ndarray]],
    distance_threshold: float,
) -> Tuple[str, float]:
    best_slug: Optional[str] = None
    best_dist = float("inf")

    for slug, emb_list in gallery_embeddings.items():
        if not emb_list:
            continue
        min_dist = min(_cosine_distance(query_embedding, gallery_emb) for gallery_emb in emb_list)
        if min_dist < best_dist:
            best_dist = min_dist
            best_slug = slug

    if best_slug is None:
        return UNKNOWN_LABEL, float("inf")

    if best_dist > distance_threshold:
        return UNKNOWN_LABEL, best_dist

    return CHARACTERS[best_slug], best_dist


def annotate_video(
    input_video: str,
    output_video: str,
    gallery_dir: str,
    distance_threshold: float,
    iou_threshold: float,
    vote_window: int,
    track_ttl_frames: int,
    max_frames: Optional[int] = None,
    fourcc: str = "mp4v",
    detect_every: int = 1,
) -> None:
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not os.path.isdir(gallery_dir):
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")

    print("[INFO] Precomputing gallery embeddings (Facenet512)...")
    gallery_embeddings = _precompute_gallery_embeddings(gallery_dir)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(output_video, fourcc_code, fps, (frame_w, frame_h))
    if not out.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_video}")

    tracks: List[Track] = []
    next_track_id = 1
    frame_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = total_frames
    if max_frames is not None:
        frames_to_process = min(max_frames, total_frames)

    detect_every = max(1, detect_every)
    detect_runs = (frames_to_process + detect_every - 1) // detect_every
    est_sec = detect_runs * 2.5  # rough CPU seconds per RetinaFace frame
    est_min = max(1, int(est_sec / 60))
    print(
        f"[INFO] Processing {frames_to_process} frames "
        f"(detect+recognize every {detect_every} frame(s); ~{est_min}+ min on CPU)..."
    )
    pbar = tqdm(total=frames_to_process, unit="frame", desc="annotate", miniters=1)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        pbar.update(1)
        pbar.set_postfix(tracks=len(tracks), refresh=True)

        run_detection = (frame_idx - 1) % detect_every == 0

        # Drop stale tracks every frame (TTL still advances on skipped frames).
        tracks = [t for t in tracks if (frame_idx - t.last_seen_frame_idx) <= track_ttl_frames]

        if run_detection:
            face_objs = DeepFace.extract_faces(
                img_path=frame_bgr,
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
                color_face="bgr",
            )

            detections: List[Tuple[Tuple[int, int, int, int], str]] = []

            for face_obj in face_objs:
                if not _is_plausible_detection(face_obj, frame_w, frame_h):
                    continue
                facial_area = face_obj.get("facial_area", {})
                x = int(facial_area.get("x", 0))
                y = int(facial_area.get("y", 0))
                w = int(facial_area.get("w", 0))
                h = int(facial_area.get("h", 0))
                bbox_xyxy = _safe_xyxy(x, y, w, h, frame_w, frame_h)

                face_img_bgr = face_obj.get("face")
                if face_img_bgr is None:
                    continue

                query_emb = _extract_embedding_facenet512(face_img_bgr)
                display_label, _dist = _best_of_character_match(
                    query_embedding=query_emb,
                    gallery_embeddings=gallery_embeddings,
                    distance_threshold=distance_threshold,
                )
                detections.append((bbox_xyxy, display_label))

            used_track_ids = set()
            used_det_indices = set()
            det_to_track: Dict[int, int] = {}

            for det_idx, (det_bbox, _det_label) in enumerate(detections):
                best_iou = 0.0
                best_track: Optional[Track] = None
                for tr in tracks:
                    if tr.track_id in used_track_ids:
                        continue
                    iou = _bbox_iou_xyxy(det_bbox, tr.bbox_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_track = tr

                if best_track is not None and best_iou >= iou_threshold:
                    used_track_ids.add(best_track.track_id)
                    used_det_indices.add(det_idx)
                    det_to_track[det_idx] = best_track.track_id

            for det_idx, (det_bbox, det_label) in enumerate(detections):
                if det_idx in used_det_indices:
                    track_id = det_to_track[det_idx]
                    for tr in tracks:
                        if tr.track_id == track_id:
                            tr.bbox_xyxy = det_bbox
                            tr.last_seen_frame_idx = frame_idx
                            tr.labels_window.append(det_label)
                            break
                else:
                    new_track = Track(
                        track_id=next_track_id,
                        bbox_xyxy=det_bbox,
                        last_seen_frame_idx=frame_idx,
                        labels_window=deque([det_label], maxlen=vote_window),
                    )
                    tracks.append(new_track)
                    next_track_id += 1

        # Draw active tracks (on detect frames: fresh boxes; on skip frames: last known).
        for tr in tracks:
            vote_label = _majority_vote(tr.labels_window)
            x1, y1, x2, y2 = tr.bbox_xyxy

            # Deterministic color from track id.
            rng = np.random.default_rng(tr.track_id)
            color = tuple(int(c) for c in rng.integers(low=0, high=255, size=3))

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color=color, thickness=2)

            text = vote_label
            # Put text slightly above the top-left corner if possible.
            text_x = x1
            text_y = max(0, y1 - 8)
            cv2.putText(
                frame_bgr,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        out.write(frame_bgr)

    pbar.close()
    cap.release()
    out.release()
    print(f"[INFO] Done. Wrote: {output_video}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate video with face boxes + character names.")
    parser.add_argument("--input", required=True, help="Input video path (e.g. input.mp4)")
    parser.add_argument("--output", required=True, help="Output annotated video path (e.g. output.mp4)")
    parser.add_argument("--gallery", required=True, help="Gallery dir with per-character subfolders")

    parser.add_argument("--distance-threshold", type=float, default=0.35, help="Cosine distance cutoff")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for track association")
    parser.add_argument("--vote-window", type=int, default=11, help="Majority vote window size (frames)")
    parser.add_argument("--track-ttl", type=int, default=15, help="How many frames a track can go missing")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug: stop after N frames")
    parser.add_argument("--fourcc", type=str, default="mp4v", help="OpenCV VideoWriter fourcc (e.g. mp4v)")
    parser.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="Run RetinaFace every N frames (default 1). Use 3–5 on CPU for ~3–5x speedup; boxes update less often.",
    )

    args = parser.parse_args()

    annotate_video(
        input_video=args.input,
        output_video=args.output,
        gallery_dir=args.gallery,
        distance_threshold=args.distance_threshold,
        iou_threshold=args.iou_threshold,
        vote_window=args.vote_window,
        track_ttl_frames=args.track_ttl,
        max_frames=args.max_frames,
        fourcc=args.fourcc,
        detect_every=args.detect_every,
    )


if __name__ == "__main__":
    main()

