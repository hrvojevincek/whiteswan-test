# White Swan — ML Engineer Coding Assessment Plan

**Task:** Face detection + recognition on a video; draw bounding boxes; label with character names when possible.

**Video:** [Google Drive](https://drive.google.com/file/d/1CM1IWUN59ZWml9MwgrvSHXz_9AirIiuU/view?usp=sharing)

**Stack:** [DeepFace](https://github.com/serengil/deepface) — **RetinaFace** (detector) + **Facenet512** (recognition)

**Characters to label:**

- Harry Potter
- Ron Weasley
- Hermione Granger
- Prof. McGonagall
- Prof. Severus Snape

**Submission:** Annotated output video + Python code + run instructions. Reasonable accuracy is enough (not perfect).

> **For agents:** Read this file before implementing. Update the progress checklist as steps complete. Do not change locked architecture decisions without user approval.

---

## Progress checklist (main plan)

- [ ] Download input video → `input.mp4` (local, gitignored if large)
- [ ] Create `gallery/` (manual face crops per character slug)
- [ ] Implement gallery embedding precompute (Facenet512, `enforce_detection=False`)
- [ ] Implement `annotate_video.py` (detect → match → track → vote → draw)
- [ ] Implement `requirements.txt` + `README.md`
- [ ] Optional: `build_gallery.py` helper
- [ ] Tune CLI thresholds on 2–3 anchor frames
- [ ] Generate `output_annotated.mp4`
- [ ] Package submission (code, `gallery/`, output video)

---

## Architecture (locked decisions)

| Area | Decision |
|------|----------|
| Identity | Reference gallery built from the same video |
| Enrollment | Manual scrub → tight face crops in per-character folders |
| Matching | Precomputed Facenet512 embeddings + cosine distance |
| Per-character score | **Best-of** — min distance across all images for that character |
| Frames | Process **every** frame |
| Uncertainty | Distance threshold → label `Unknown` (do not force closest name) |
| Temporal stability | Majority vote per face track; `Unknown` **counts** in the vote |
| Tracking | IoU association between frames; **inactive TTL** (~15 frames) before dropping track |
| Models | RetinaFace + Facenet512 for video; gallery embed with `enforce_detection=False` |
| Repo layout | Two scripts: optional `build_gallery.py` + main `annotate_video.py` |
| Runtime | CPU-first; GPU optional (document in README) |
| Output video | OpenCV `VideoWriter` → `.mp4` |
| Tunables | CLI flags: distance threshold, IoU, vote window, track TTL |
| Submission | Include `gallery/` + `output_annotated.mp4` |
| Display names | Slug folders + `CHARACTERS` map to exact assessment spellings |

---

## Pipeline

```
OFFLINE
  Video → manual crops → gallery/{slug}/*.jpg
       → DeepFace.represent (Facenet512, enforce_detection=False)
       → embedding index per image

ONLINE (every frame)
  Frame → RetinaFace → embed each face (Facenet512)
       → best-of-C cosine vs gallery → threshold → raw label (name | Unknown)
       → IoU tracks (+ inactive TTL) → majority vote (Unknown counts)
       → draw box + CHARACTERS[slug] → output_annotated.mp4
```

---

## Repository layout (target)

```
whiteswan/
├── PLAN.md                 # this file — source of truth
├── gallery/
│   ├── harry/
│   ├── ron/
│   ├── hermione/
│   ├── mcgonagall/
│   └── snape/
├── build_gallery.py        # optional helper
├── annotate_video.py       # main entrypoint
├── requirements.txt
├── README.md
├── input.mp4               # local only (gitignore if large)
└── output_annotated.mp4    # submit with assessment
```

---

## Character map

```python
CHARACTERS = {
    "harry": "Harry Potter",
    "ron": "Ron Weasley",
    "hermione": "Hermione Granger",
    "mcgonagall": "Prof. McGonagall",
    "snape": "Prof. Severus Snape",
}
```

Gallery folders use **slugs**; drawn labels use **display names** from this map.

---

## CLI (target)

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

Tune `--distance-threshold` on 2–3 frames where ground truth is known. Document distance vs similarity in README.

---

## README must cover

1. Python version (e.g. 3.10–3.11) and `pip install -r requirements.txt`
2. First-run model download note
3. How to build `gallery/` (manual crops; optional helper script)
4. Exact run command for `annotate_video.py`
5. CPU vs GPU note
6. MP4 / FourCC note if playback issues on macOS/Linux

---

## Known limitations (acceptable for submission)

- Similar adult faces (McGonagall / Snape) may confuse without diverse gallery crops
- Extra faces / partial faces → `Unknown` (matches “when possible”)
- Every frame + RetinaFace on CPU can be slow
- Labels may flicker if threshold/window poorly tuned

---

## Revisit after main plan (polish & hardening)

Complete the main checklist first, then work through these in order. Check off as done.

### Accuracy & gallery

- [ ] **Gallery quality pass** — Add 2–3 more crops per character for hard angles (profile, dim light, hats/glasses); remove blurry duplicates
- [ ] **Per-character threshold** — If one character confuses another often, try separate distance thresholds per slug (optional CLI)
- [ ] **Anchor-frame evaluation** — Scrub 5–10 known frames; count correct / wrong / Unknown; log results in README or `EVAL.md`
- [ ] **Adult confusion (McGonagall / Snape)** — Extra gallery diversity or slightly stricter threshold for those slugs only

### Matching implementation details

- [ ] **L2-normalize embeddings** — Confirm cosine uses normalized Facenet512 vectors (DeepFace output may already be normalized; verify once)
- [ ] **Tie-breaking** — When two characters are within epsilon distance, prefer `Unknown` over guessing
- [ ] **Distance metric docs** — README states exact formula (cosine distance vs similarity) and how threshold was chosen

### Video & stability

- [ ] **Threshold / vote-window sweep** — Quick grid on anchor frames if flicker or wrong names persist
- [ ] **Track TTL tuning** — If labels reset on brief occlusions, increase `--track-ttl`; if tracks merge two people, decrease it
- [ ] **IoU threshold tuning** — If IDs swap between adjacent faces, raise `--iou-threshold`
- [ ] **MP4 playback** — If output won’t play on reviewer machine, try alternate FourCC or ffmpeg re-encode path (document in README)

### Performance (only if needed)

- [ ] **Runtime benchmark** — Note wall-clock time for full video on CPU (README)
- [ ] **GPU path** — Verify TensorFlow GPU path and document speedup if substantial
- [ ] **Frame skip fallback** — Only if CPU runtime unacceptable: detect every frame, recognize every Nth (document tradeoff)

### Submission polish

- [ ] **Gallery provenance** — `gallery/README.md` with approximate timestamps per slug for each crop
- [ ] **`.gitignore`** — Ignore `input.mp4`, `__pycache__`, `.venv`, model cache if huge
- [ ] **Final watch-through** — Full play of `output_annotated.mp4`; note known failure scenes in README (honest scope)
- [ ] **Zip / deliverable check** — Code, `requirements.txt`, README, `gallery/`, `output_annotated.mp4`; input video not required unless asked

### Optional extras (only if time)

- [ ] **Per-character box colors** — Visual distinction in crowded scenes
- [ ] **Progress bar** — `tqdm` over frames for long CPU runs
- [ ] **`build_gallery.py`** — Frame scrub helper to save crops with keyboard shortcuts
- [ ] **K consecutive named matches** — Stricter label flip than majority vote if flicker remains

---

## Open implementation notes

- Use same `model_name='Facenet512'` and `detector_backend='retinaface'` for video inference; gallery uses `enforce_detection=False`
- Majority vote deque length = `--vote-window`; inactive tracks kept `--track-ttl` frames without match
- Do not commit secrets or personal paths in README

---

## Session log

| Date | Notes |
|------|-------|
| 2026-05-23 | Plan finalized via design review (grill-me). Architecture locked. `PLAN.md` created. Implementation not started. |
