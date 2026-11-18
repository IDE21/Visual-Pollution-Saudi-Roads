# External pothole benchmark

The dataset paper (AlElaiwi et al., 2022) re-tested its model on the Roboflow "Pothole Detection Dataset". That is the only number in the paper we can reproduce on exactly the same images, so it is the fairest head-to-head comparison available.

## Data

- **Source:** [Pothole Detection Dataset](https://universe.roboflow.com/aegis/pothole-detection-i00zy) by Aegis, Roboflow Universe, **version 2** (1,482 images, no augmentation). Licence: **CC BY 4.0**.
- Later versions (v3–v10, about 3,500 images) add rotated and brightness-shifted copies, so v2 is the one that matches the paper's image count.
- All splits were pooled: 1,482 images and 5,014 pothole boxes. 144 images have no pothole.
- Prepared with [`src/prep_external_pothole.py`](../../src/prep_external_pothole.py), which remaps class 0 to our pothole id (2). The images are not redistributed here.

## Protocol

- Model: v1 (YOLO11s, 640 px, trained on our clean split), with **no fine-tuning** on this dataset.
- Inference at 640 px. Only the pothole class is scored. Barrier and sidewalk predictions cannot match a pothole box, so they do not affect the pothole numbers.
- Metrics from [`src/eval_ci.py`](../../src/eval_ci.py): mAP50, mAP50-95, and P/R/F1 at the confidence that maximizes F1 (IoU 0.5).
- 95% CIs from 2,000 bootstrap resamples **by image**: the dataset has no incident ids. Neighbouring video frames may be correlated, so the true CI is probably wider.

## Results

| | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| **v1 (ours), no fine-tuning** | 0.631 | 0.487 | 0.550 | **0.551** [0.536, 0.568] | 0.271 [0.260, 0.282] |
| Paper (YOLOv5x) | 0.81 | 0.75 | — | 0.70 (type not stated) | — |
| v1 on our own clean test set C, potholes only | — | — | — | 0.794 | 0.549 |

## Findings

1. **On this benchmark, the paper's model is better.** Its 70% mAP is about 15 points above our 55% mAP50. Its precision and recall are also higher. The paper does not state its mAP type or whether it fine-tuned on this dataset, so the gap is indicative, not exact.
2. **Our model drops sharply out of domain:** pothole mAP50 falls from 0.79 on our clean test set to 0.55 here.
3. **The two datasets label potholes very differently:**

   | | Boxes per image with potholes | Median box area (% of image) |
   |---|---|---|
   | Our clean test set C | 1.2 | 4.9% |
   | External set | 3.75 | 0.9% |

   External potholes are about 5× smaller and 3× denser. The images are also small (640×352 or 416×416). A model trained mostly on large, single potholes misses many small ones, which matches the low recall (0.49).
4. **Implication for the plan:** we expected higher input resolution and small-object handling to help here. The follow-up below tested this, and they did not.

## Follow-up: can resolution or slicing close the gap?

The domain-shift finding suggested that small potholes are the problem, so we tried the cheap fixes first, all **without training**: larger inference sizes, sliced inference (SAHI, [`src/predict_sahi.py`](../../src/predict_sahi.py)), and the two resolution-screening models (half the training data, 40 epochs, see the [screening report](../resolution_screen/report.md)). All rows are on the same 1,482 images. The differences are paired against v1 at 640 px, with 95% CIs by image.

| Model | Inference | P | R | F1 | mAP50 | mAP50-95 | Δ mAP50 vs v1 |
|---|---|---|---|---|---|---|---|
| **v1** (full data, 50 epochs) | 640 px | 0.631 | 0.487 | 0.550 | **0.551** | **0.271** | — |
| v1 | 960 px | 0.565 | 0.488 | 0.523 | 0.518 | 0.240 | −0.034 [−0.044, −0.024] |
| v1 | 1280 px | 0.543 | 0.421 | 0.474 | 0.439 | 0.189 | −0.112 [−0.125, −0.099] |
| v1 | SAHI, 320 px slices | 0.569 | 0.466 | 0.512 | 0.496 | 0.235 | −0.055 [−0.066, −0.045] |
| v1 | SAHI, 256 px slices | 0.520 | 0.431 | 0.471 | 0.444 | 0.206 | −0.107 [−0.119, −0.095] |
| Screening model, trained at 640 | 640 px | 0.582 | 0.434 | 0.497 | 0.476 | 0.228 | — |
| Screening model, trained at 960 | 960 px | 0.598 | 0.421 | 0.494 | 0.481 | 0.239 | — |

Paired, trained at 960 minus trained at 640: mAP50 +0.005 [−0.006, +0.015], mAP50-95 +0.011 [+0.006, +0.017].

**Findings**

1. **Nothing cheap closes the gap.** Every inference-time change made v1 worse. Training at 960 px adds about one mAP50-95 point here, the same small pothole-only gain seen on our own val.
2. **Upscaling hurts.** The external images are only 640×352 or 416×416, so larger inference sizes and small slices just enlarge blurry pixels. The model then sees potholes at scales it never saw in training.
3. **More training data helped far more than resolution.** The screening models (half the data, 40 epochs) score about 7 mAP50 points below v1 (full data, 50 epochs). Both data and epochs changed, so we cannot split the two effects.
4. **So the gap is mostly a difference in what counts as a pothole**, not a resolution problem. The external labels mark many small, dense potholes, and the paper may also have fine-tuned on this set (it does not say). Matching the paper here would need fine-tuning on some external images, which would no longer be a zero-shot test.

## Limits

- One model, one seed, no fine-tuning.
- The paper's external-test protocol (mAP type, IoU, confidence threshold, fine-tuning) is not documented, so the comparison is approximate.
- The CIs are per image, not per scene, and are probably too narrow.
