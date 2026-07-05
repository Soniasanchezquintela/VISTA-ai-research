<!--
  REPORT SKELETON (PRELIMINARY DRAFT) — not the final report.
  Hypothesis & Setup are filled in as a guideline; verify against your actual work.
  RESULTS sections contain [PLACEHOLDER] — fill with REAL numbers/figures. Do NOT
  ship invented metrics. Delete these HTML comments before submitting.
  When ready, this content moves into README.md (the graded Final Report).
-->

# VISTA — [project tagline, e.g. "Assistive shelf navigation for visually impaired shoppers"]

> 🔗 **Repository:** https://github.com/Soniasanchezquintela/VISTA-ai-research
> (Link also available in the PPT presentation).
> 
“Accessibility if not about convenience, 
It is about dignity”. Improving the shopping experience for visually impaired people.

**Team Members:** Ramon Viedma, Sonia Sánchez, Nuria Olvera, Petros Zonias
**Advisor:** Amanda Duarte

---

## 1. Overview

**Social Motivation/Impact.** For a sighted person, shopping in a supermarket can be a simple 20-minute task. For a 
blind or visually impaired person, finding a product on a shelf can require depending on someone else being available
to assist. This projects aims to reduce that dependence.With the VISTA project we want to give users more autonomy, confidence, and independence for the day shopping tasks. It also can allow shops/retailers implement social safeguards using AI for Good for special collectives like teh visually impaired.

**Goal.** VISTA is a prototype assistive system that uses a single camera and the
user's voice to (a) describe the products on a shelf, and (b) tell the user what
product they are pointing at — spoken back in their language.

**End-to-end user story.**
1. The user faces a shelf wearing/holding the camera and speaks a command.
2. The system transcribes the speech and classifies the *intent*
   (e.g. "describe the scene" or "what am I pointing at?").
3. It detects and identifies the products in view, tracks them across frames,
   and resolves which one the finger is pointing at.
4. It speaks back a concise, useful answer.

**Scope of this report.** This report presents the structure of our project, how we built each module, and the results of the experiments we conducted.

### System architecture
<img src="images/Screenshot 2026-07-05 at 17.45.04.png" width="700" alt="YOLO training and validation curves">
```
Voice (wake word → speech-to-text → intent)
        │
        ▼
Camera frame ──► Product detection (YOLO) ──► Product identification (CLIP)
        │                                              │
        └──► Hand / pointing detection (MediaPipe) ──► Scene memory (tracking)
                                                       │
                                                       ▼
                                          Spoken response (scene / pointed product)
```

| Component | Module | Trained by us? | Status |
|---|---|---|---|
| Product detection | `object_detector/` | ✅ Yes (fine-tuned YOLO) | [ ] |
| Product identification | `object_identifier/` | ❌ No (frozen CLIP, zero-shot) | [ ] |
| Hand & pointing | `hand_detector.py` | ❌ No (pretrained MediaPipe) | [ ] |
| Scene memory / tracking | `scene_memory/` | ❌ No (classical algorithm) | [ ] |
| Voice → intent | `voice/`, `intent_classifier/` | ✅ Yes (fine-tuned DistilBERT) | [ ] |
| Description | `scene_memory/`, [LLM] | ❌ No (prompt-based) | [ ] |

---

## 2. How to run the code

### Requirements
- Python **3.10+** (the codebase uses `X | None` type syntax at runtime).
- [OS / GPU notes.]

### Setup
```bash
git clone https://github.com/Soniasanchezquintela/VISTA-ai-research.git
cd VISTA-ai-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Model files (not in the repo — download separately)
| File | Put in | Source |
|---|---|---|
| `sku110k_768_e20_pat5.pt` (YOLO weights) | `object_detector/` | [Drive link] |
| `hand_landmarker.task` (MediaPipe) | project root | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task |
| `intent_classifier.pt` (intent model) | `intent_classifier/checkpoints/` | [Drive link, or train via `python -m intent_classifier.train`] |

### Running
```bash
python project.py --image path/to/shelf.jpg   # single image
python project.py --webcam 0                   # live; type `listen` for a voice command
```

### Tests
```bash
python test_scene_memory.py
```

---

## 3. Experiments

> Each subsection: **Hypothesis → Setup → Results → Conclusions.**
> Note which are *training* experiments (YOLO, intent classifier) vs *evaluation /
> tuning / integration* experiments (CLIP, MediaPipe, scene memory, description).

---

### Experiment 1 — Fine-tuning a detector for dense shelf products (YOLO / SKU110K)
*Type: neural-network training experiment.*

**Hypothesis**
A general-purpose - out of the box - YOLO model will not, out of the box, reliably localize the
small, densely packed, repeated products found on supermarket shelves. We expect
that fine-tuning YOLO11 on SKU110K (a dataset built specifically for dense retail
shelves) will substantially improve detection recall on our own shelf photos,
and that a larger input resolution will help most because the products are small
relative to the frame.

**Experiment setup**
- **Base model:** `yolo11m.pt` (Ultralytics YOLO11-medium).
- **Data:** SKU110K (`sku110k_product.yaml`), a single "product" class of densely
  packed retail items. [State train/val/test counts.]
- **Training config:** `epochs=50`, `patience=5` (early stopping), `imgsz=768`,
  `batch=8`. [Optimizer/LR = Ultralytics defaults unless changed.]
- **Tuning explored:** input size 640 vs 768, batch 8 vs 4 (see throughput notes
  in `train_yolo.py`); the shipped weights are `imgsz=768`, early-stopped (the
  filename `sku110k_768_e20_pat5` reflects ~20 effective epochs, patience 5).
- **Evaluation:** [mAP@0.5 / mAP@0.5:0.95 on SKU110K val] + qualitative results on
  our own Ametller/iPhone shelf photos. Artifacts in `object_detector/train_yolo_results/`.

**Results**
- **mAP@0.5 ≈ 0.92** — at an IoU threshold of 0.50, the detector localizes shelf
  products with high precision/recall.
- **mAP@0.5:0.95 = 0.58** — averaged over stricter IoU thresholds (0.50–0.95),
  the score drops, indicating boxes are found reliably but not always tightly
  aligned to the product edges.
- **Training curves** (below): all training and validation losses (box, cls, dfl)
  decrease smoothly and converge, with no divergence between train and val — no
  sign of overfitting. Precision (~0.92), recall (~0.88), mAP@0.5 (~0.92) and
  mAP@0.5:0.95 (~0.58) all rise and plateau over ~45 epochs.

  <img src="images/Yolo-results.png" width="700" alt="YOLO training and validation curves">

  *Figure 1 — YOLO11 training/validation losses and detection metrics over epochs.*
- [PLACEHOLDER — 2–3 annotated sample images on our own photos.]
- [PLACEHOLDER — 640 vs 768 comparison if run.]

**Conclusions**
[What the numbers show: did fine-tuning + higher resolution help? Failure modes —
tiny products, occlusion, glare, top/bottom shelf rows. Did early stopping trigger
before 50 epochs (overfitting risk on a single class)? New hypotheses for next round.]

---

### Experiment 2 — Zero-shot product identification with CLIP + a reference catalog
*Type: evaluation / threshold-tuning experiment (no training).*

**Hypothesis**
Rather than training a classifier per product (which doesn't scale as the catalog
grows), we expect a frozen CLIP encoder plus a small catalog of reference images
to identify a cropped product by nearest-neighbour in embedding space — and that
confidence thresholds can suppress wrong guesses on products not in the catalog.

**Prior approach and why we changed**
Our first attempt was **image-to-text**: we fine-tuned the last two layers of a
CLIP model so that a bounding-box crop's image embedding would align with the
**text embedding of the product label**, and identified a product by matching the
crop against those label embeddings. Evaluated over a catalog of **~1000 products**,
this reached **recall@1 ≈ 70%**.

The decisive problem, however, was **not** the aggregate recall figure — it was
that specific products were **consistently, reproducibly mismatched**, not randomly
wrong. Two linked weaknesses drove this:

1. **Language dependency.** Product labels were in Spanish/Catalan, while CLIP's
   text encoder is trained predominantly on English, so labels had to be translated
   to English to match at all — an extra, fragile step in the pipeline.
2. **Weak, coarse text matching even after translation.** Similarity scores stayed
   low (~0.26–0.34) and semantically-adjacent products were reliably confused. For
   example (query shown as `Spanish label → English translation`, then top matches
   with cosine scores):

   ```
   'yogur natural' → 'natural yogurt'
     [0.314] Yogur natural de cabra Hacendado — Postres y yogures
     [0.313] Queso fresco Burgos natural Hacendado — Charcutería y quesos   ← fresh CHEESE, not yogurt
     [0.291] Queso fresco Burgos natural Hacendado — Charcutería y quesos

   'pan de molde' → 'sliced bread'
     [0.337] Barra pan de pueblo rebanada — Panadería y pastelería          ← a baguette-style loaf,
     [0.322] Barra de pan espiga rebanado — Panadería y pastelería             not sandwich bread
     [0.321] Barra de pan campesina masa madre rebanada — Panadería y pastelería
   ```

   The matches are near-ties at low confidence, so the "winner" for these items was
   effectively arbitrary among a cluster of wrong-but-related products.

For an assistive use case, a system that *reliably* fails on specific products is
worse than one that fails randomly: the user learns it "can never find" those items.

We therefore switched to the **image-to-image** method described below: comparing the
crop's image embedding directly to image embeddings of catalog photos. This removes
text — and therefore the translation step and language dependency — from the pipeline
entirely, and compares like-with-like (photo vs photo) instead of photo vs label. It
is also zero-shot (no fine-tuning) and scales simply by adding more reference images.

**Experiment setup**
- **Model:** `open_clip` ViT-B-32 (laion2b), frozen — no fine-tuning.
- **Catalog:** [N] reference products (currently **10** SKUs, mostly plant-based
  milks, scraped from Ametller; embeddings precomputed in
  `product_db/embeddings/`). Metadata in `products.sqlite`.
- **Matching:** cosine similarity of the crop embedding vs catalog embeddings;
  accept if `score ≥ MIN_SCORE (0.70)` and `confidence ≥ MIN_CONFIDENCE (0.80)`;
  recent change: when no match clears 0.70, the system returns the top candidate
  options instead of a single answer.
- **Evaluation:** [held-out crops of catalog products + distractor products;
  report top-1 accuracy and false-accept rate on unknowns.]

**Results**
- [PLACEHOLDER — top-1 identification accuracy on catalog products.]
- [PLACEHOLDER — false-accept rate on out-of-catalog products vs threshold.]
- [PLACEHOLDER — examples of correct match vs confusion (similar packaging).]

**Conclusions**
[Effect of tiny catalog (only similar products → easy to confuse); threshold
trade-off (recall vs false accepts); whether zero-shot CLIP is good enough or
needs fine-tuning / more reference images per SKU. New hypotheses.]

---

### Experiment 3 — Pointing-based product selection (MediaPipe Hands)
*Type: integration / evaluation experiment (pretrained model).*

<img width="447" height="252" alt="Screenshot 2026-07-03 at 14 56 52" src="https://github.com/user-attachments/assets/a14f51ee-3195-4d6e-a063-c9f4e40b446e" />
<img width="446" height="245" alt="Screenshot 2026-07-03 at 14 59 55" src="https://github.com/user-attachments/assets/6f5f82fc-5797-4e28-8d52-ba0f8c2af064" />




**Hypothesis**
We expect index-finger landmarks from a pretrained hand detector to give a
pointing signal precise enough to select which detected product a user means,
without any custom training.

**Experiment setup**
- **Model:** MediaPipe HandLandmarker (`hand_landmarker.task`, float16), in image
  and video modes (`hand_detector.py`).
- **Pointing logic:** [how the finger tip / direction is computed and mapped to a
  detected box — e.g. tip inside box, else nearest box within tolerance].
- **Evaluation:** [point at a known target across M trials; report % of trials the
  intended product was selected; sensitivity to distance/angle/lighting].

**Results**
- [PLACEHOLDER — selection success rate.]
- [PLACEHOLDER — screenshots of correct/incorrect selection.]

**Conclusions**
[Reliability, where it breaks (hand occludes product, multiple boxes overlap),
interaction with scene memory. New hypotheses.]

---

### Experiment 4 — Persistent scene memory vs per-frame detection (tracking)
*Type: algorithmic / ablation experiment (no NN).*

**Hypothesis**
A per-frame pipeline that re-detects and re-identifies from scratch each frame
will produce unstable output (flickering boxes, changing IDs) and cannot support
pointing, because nothing persists between frames. We expect that adding a
tracking layer — matching boxes across frames, remembering them briefly when
occluded, and voting on their identity over time — will give stable identities,
survive a hand passing over a product, stay stable when the camera pans, and make
pointing usable.

**Experiment setup**
- **Baseline:** original `ShelfSceneMemory` — resets every frame;
  `get_touched_object()` returns `None` (pointing impossible).
- **Ours:** `TrackedShelfSceneMemory`. The public interface is unchanged, so it
  is a drop-in replacement. It combines three mechanisms (detailed below):
  IoU matching, product voting, and global-motion compensation.
- **Evaluation:** unit tests (`test_scene_memory.py`: ID stability,
  new-vs-existing, occlusion survival/removal, pointing hit/miss, voting,
  promotion, bbox smoothing) + qualitative webcam runs comparing baseline vs ours.

**Core components**

*(1) IoU (Intersection over Union) — the matching mechanism.*
IoU measures how much two boxes overlap, relative to their combined area
(0 = disjoint, 1 = identical). Each frame, every remembered product is matched to
the new detection it overlaps most; overlap **≥ 30%** counts as "the same product"
(keep its ID and history), below 30% does not. The 30% threshold balances two
failure modes: too high loses identity on small movements, too low merges
neighbouring products.

*(2) Product voting — the identification mechanism.*
A single CLIP reading can be wrong (blur, glare, angle). Instead of trusting one
frame, each track keeps a per-identity **vote tally**: every confident frame adds
its score, and the displayed label is whichever identity leads. A single bad frame
therefore cannot override a repeated, confident answer. Low-confidence frames cast
no vote (the system stays silent rather than guess); votes reset if a track loses
its identity. This is what makes the green "recognized" label trustworthy.

*(3) Motion compensation — robustness to camera movement.*
Because matching relies on IoU, a camera pan (the user turning their head) moves
every box at once and would break every match — leaving greyed "ghost" boxes and
renumbering everything. Insight: a pan shifts every product by the *same* vector.
We recover that vector by **displacement voting** — measuring how far each old box
would move to reach each new box, and taking the shift the most products agree on
(robust to large shifts, occluded products, and one product moving on its own). We
then shift old boxes to their predicted positions before matching, so identities
survive the pan. Limitation: this models translation only, not large rotation/zoom.

*Occlusion persistence.* A product that disappears (e.g. a hand covering it) is
kept as a greyed "lost" track for up to **90 frames (~3 s)** before removal, then
revived with its original ID if it reappears — enough time for a slow, deliberate
hand placement.

**Results**
- Unit tests: [PLACEHOLDER — "N/N passing"] covering ID stability, occlusion
  survival, pointing resolution, voting, and bbox smoothing.
- [PLACEHOLDER — before/after webcam: box-flicker / ID-switch count over a fixed
  clip; box survival time under hand occlusion.]
- [PLACEHOLDER — head-turn comparison: ghost-box count with vs without motion
  compensation.]
- [PLACEHOLDER — screenshot of a product keeping its ID while briefly covered
  (grey "lost" box → revived), and green boxes when CLIP recognizes a product.]

**Conclusions**
Tracking is what makes pointing function at all (the baseline always returns
`None`). IoU matching + 90-frame persistence give stable IDs and survive brief
occlusion; voting stabilises the identity label; motion compensation removes the
ghost-box artefact on head turns. Remaining failure modes: very fast *independent*
motion of a single product (an *ID switch* — the box is treated as old-gone +
new-arrived, losing its vote history), and large rotation/zoom (not captured by
translation-only compensation). Future work: motion prediction (Kalman/SORT-style)
or appearance-based re-linking to survive these cases.

---

### Experiment 5 — Voice command understanding (Whisper STT + intent classifier)
*Type: neural-network training experiment (the intent classifier is fine-tuned).*

**Hypothesis**
We expect that off-the-shelf speech-to-text plus a fine-tuned multilingual intent
classifier can map spoken commands (Spanish/Catalan) to the correct action, and
that fine-tuning a compact transformer on a small hand-built intent dataset is
enough for the limited command vocabulary of this application.

**Experiment setup**
- **STT:** `faster-whisper` (base model, Spanish), via `voice_to_text.py`.
- **Intent model:** `distilbert-base-multilingual-cased` encoder + MLP head
  (`hidden_size=256`, `dropout=0.2`), fine-tuned with AdamW, `max_length=64`, a
  validation split, and a confidence threshold that falls back to `unknown`.
- **Data:** `intent_classifier/data/intents.csv` — ~**202** labelled commands
  across **7** intents (describe_scene, describe_pointed_product,
  navigate_to_target, confirm_target_present, get_price, read_text, unknown),
  plus rule-based target extraction.
- **Evaluation:** [validation accuracy per intent; target-extraction accuracy;
  end-to-end voice→intent on live mic samples].

**Results**
- [PLACEHOLDER — validation accuracy + per-intent confusion matrix.]
- [PLACEHOLDER — training/val accuracy curve from `checkpoints/`.]
- [PLACEHOLDER — end-to-end examples: spoken phrase → transcription → intent.]

**Conclusions**
[Which intents are reliable vs confused (small dataset risk); how STT errors
cascade into wrong intents; whether 202 examples / 7 classes is enough or the
dataset needs expansion; Spanish vs Catalan coverage. New hypotheses.]

---

### Experiment 6 — Natural-language description (scene & pointed product)
*Type: integration experiment (prompt-based / templated, no training).*

**Hypothesis**
We expect that a concise, rule-based or LLM-generated description of the scene and
of the pointed product gives a visually impaired user genuinely useful spoken
information, and that grouping duplicates / counting unknowns avoids overwhelming
them.

**Experiment setup**
- **Templated baseline:** `describe_scene()` / `describe_pointed_product()` —
  group identical SKUs, count unknowns, output Spanish sentences.
- **LLM option:** [Ollama VLM — Gemma3/Qwen — in `ask_qwen.py`] for richer
  descriptions. [State which is used in the live system.]
- **Inputs:** visible tracks from scene memory; the touched track for pointing.
- **Evaluation:** [qualitative — sample outputs rated for clarity/usefulness;
  note any hallucinations from the LLM path].

**Results**
- [PLACEHOLDER — sample scene description outputs.]
- [PLACEHOLDER — sample pointed-product outputs (known vs unknown product).]

[Template vs LLM trade-off (reliability vs richness); hallucination risk; what a
real user would actually need. New hypotheses.]

---

## ❇️ 4. Overall conclusions & future work

[Synthesize across experiments: what works end-to-end today, the weakest link
(e.g. the 10-SKU catalog limiting identification), most promising next steps,
tie back to the assistive use case.]
🚀 Overall, this multi-modal approach to support the visually impaired works and shows
a lot of potential. 
💡We have learned that the systems nees from than frame-by-frame AI; it needs memory,
confidence, tracking, and user-centered interaction. 
⚒️ What we would do differently: collect realistic video data earlier, design the scene 
memory layer from teh start, modularize the pipeline more clearly, work in robustness,
and evaluate sooner. 

What next steps could be: 
🛍️ Increase the supermarket dataset to have more images and a more diversified set 
of sub-contexts within the supermarket (e.g., product type, disposition, light, access).
👩‍✈️Pilot the project with visually impaired persons.

---

## 5. References

[Format consistently — e.g. numbered or author-year. Pull the related papers from
`papers.md`. Suggested entries:]

**Datasets**
- [SKU110K — Goldman et al., "Precise Detection in Densely Packed Scenes," CVPR 2019.]

**Models & libraries**
- [Ultralytics YOLO11 — detection.]
- [OpenCLIP / CLIP — Radford et al., "Learning Transferable Visual Models From
  Natural Language Supervision," 2021; open_clip ViT-B-32 (laion2b).]
- [MediaPipe Hands / HandLandmarker — Google.]
- [faster-whisper (CTranslate2) — Whisper, Radford et al., 2022.]
- [DistilBERT — Sanh et al., 2019; `distilbert-base-multilingual-cased`.]
- [Multi-object tracking background (IoU/SORT) — Bewley et al., 2016 — for the
  scene-memory section's future-work discussion.]

**Related work**
- [The 1–3 papers listed in `papers.md` that motivated the assistive use case.]

---

<!-- APPENDIX (optional) -->
## Appendix [optional]

- [Per-experiment extra figures, full hyperparameter tables, additional sample outputs.]
- [Link to slides — note: slides are a separate deliverable and do NOT count as the report.]
- Deeper dive on scene memory internals: see [`SCENE_MEMORY_EXPLAINED.md`](SCENE_MEMORY_EXPLAINED.md).
