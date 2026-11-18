# Resolution screening: 640 vs 960 px (plan v2, step 5)

**Question:** does a larger input size help enough to justify training the final model at 960 px?

## Protocol

- Script: [`scripts/screen_resolution.sh`](../../scripts/screen_resolution.sh).
- YOLO11s, pretrained. The first half of the sorted v1 train list (`fraction 0.5`), 40 epochs, batch 16, seed 42, no early stopping. Both runs used identical settings except `imgsz`.
- Each run's `best.pt` was scored on the full v1 **val** split (3,053 images, 2,790 incidents) at its own resolution. The test split was not touched.
- 95% CIs from 2,000 bootstrap resamples **by incident**. The difference is **paired**: both models are scored on the same resamples.
- Hardware: Apple M5 Pro (MPS). 960 px took about 15 min per epoch; 640 px took about 6.8 min.

## Results

| | mAP50 [95% CI] | mAP50-95 [95% CI] |
|---|---|---|
| 640 px | 0.642 [0.625, 0.662] | 0.417 [0.404, 0.432] |
| 960 px | 0.638 [0.620, 0.657] | 0.417 [0.405, 0.433] |

**Paired difference, 960 minus 640:**

| Class | mAP50 | mAP50-95 |
|---|---|---|
| barriers | −0.005 [−0.030, +0.019] | +0.003 [−0.016, +0.021] |
| sidewalks | −0.018 [−0.046, +0.009] | −0.012 [−0.030, +0.005] |
| pothole | +0.009 [−0.001, +0.020] | **+0.009 [+0.001, +0.017]** |
| **all** | **−0.005 [−0.018, +0.008]** | **+0.000 [−0.010, +0.009]** |

Per-epoch curves: [`results_640.csv`](results_640.csv), [`results_960.csv`](results_960.csv). Full metrics: [`val_640.json`](val_640.json), [`val_960.json`](val_960.json), [`val_960_minus_640.json`](val_960_minus_640.json).

## Findings

1. **No overall gain from 960 px.** The mAP50-95 difference is zero, with a CI under one point either way.
2. **Potholes are the one exception.** 960 px is slightly better, by +0.9 mAP50-95 points, and the CI excludes zero. Barriers, although small, did not benefit.
3. **960 px costs about twice as much**, in training and in inference.

**Decision:** keep 640 px for the rest of plan v2.

## Limits

- One seed per setting. The CIs cover image sampling, not training variance.
- Half the training data and 40 epochs. The ranking could change at full data and 100 epochs, but a gap above about one point is unlikely given these CIs.
