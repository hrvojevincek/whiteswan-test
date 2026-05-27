# White Swan — Face Detection + Recognition (DeepFace)

This repository contains a Python script that:

1. Detects all faces in a video (RetinaFace via `deepface`)
2. Recognizes faces against a small **reference gallery** (Facenet512 via `deepface`)
3. Draws bounding boxes and labels with the corresponding character name when possible

Accuracy does not need to be perfect; reasonable performance is sufficient for the assessment.

## Prerequisites

- Python 3.10+ recommended
- `ffmpeg` is not required (OpenCV writes MP4)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run may download model weights (RetinaFace + Facenet512). This can take a few minutes.

**Note:** TensorFlow 2.16+ requires `tf-keras` (included in `requirements.txt`). If you see `ModuleNotFoundError: No module named 'tf_keras'`, run:

```bash
pip install tf-keras
```

## Prepare the reference gallery

Create a folder `gallery/` with **one subfolder per character**. Each subfolder should contain **3–10** face crops (tight crops are better than full-frame images).

### Interactive helper (recommended)

```bash
python build_gallery.py --input input.mp4 --gallery gallery
```

**Workflow (click the video window first):**

1. **`j`** / **`n`** — find a clear frame (faces toward camera).
2. **`d`** — detect faces (green boxes; yellow = selected if several).
3. **`h`** **`r`** **`e`** **`m`** **`s`** — save face to **H**arry / **R**on / **E**rmione / **M**cGonagall / **S**nape.  
   If 2+ faces: press **`1`** **`2`** **`3`** to pick which box, then **`h`** etc.
4. **`n`** — **next frame** and repeat (this continues through the video).

Other: **`p`** prev frame · **`j`**/**`k`** skip ±30 · **`q`** quit.

Uses **RetinaFace** by default (same as `annotate_video.py`) so side/profile faces are detected more reliably. Slower on CPU — wait after **`d`**. For speed only: `--detector opencv` (misses many faces).

### Manual crops

Expected folder names (slugs) and drawn labels:

- `gallery/harry/` → `Harry Potter`
- `gallery/ron/` → `Ron Weasley`
- `gallery/hermione/` → `Hermione Granger`
- `gallery/mcgonagall/` → `Prof. McGonagall`
- `gallery/snape/` → `Prof. Severus Snape`

Notes:

- Crops should be face-centered with a small margin.
- Different angles help, especially for the adults (`McGonagall` vs `Snape`).

## Run

```bash
python annotate_video.py \
  --input input.mp4 \
  --output output_annotated.mp4 \
  --gallery gallery \
  --distance-threshold 0.35 \
  --iou-threshold 0.3 \
  --vote-window 11 \
  --track-ttl 15
```

### Tunables (what they affect)

- `--distance-threshold`: cosine distance cutoff for accepting a match. If above threshold, label is `Unknown`.
- `--iou-threshold`: IoU threshold used to associate detections to existing tracks.
- `--vote-window`: number of recent labels stored per face track; we display the majority vote.
- `--track-ttl`: how many frames a track can be missing before it is dropped.

If output labels flicker or mislabel characters, tune `--distance-threshold` first, then `--iou-threshold` and `--vote-window`.

On CPU, full-video runs are slow (~2+ s/frame). Use `--detect-every 3` (or `5`) to run RetinaFace less often — much faster; boxes/labels update every N frames instead of every frame.

## Output

The script writes `output_annotated.mp4` containing bounding boxes and the voted character label per face track.

## How identity matching works (high level)

- Gallery images are embedded once with `Facenet512`.
- For each detected face in the video, we compute an embedding and do **best-of** matching per character (minimum distance across that character’s gallery images).
- The final displayed name is stabilized over time using a per-track majority vote (including `Unknown`).
