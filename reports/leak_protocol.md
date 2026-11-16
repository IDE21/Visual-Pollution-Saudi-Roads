# Track A protocol: how much does a random split inflate the score?

Written **before** the Track A model is trained, so the analysis cannot be tuned to the result.

## Question

If this dataset is split at random at the image level, as in the dataset paper, how much higher does our model score than on a split grouped by incident? Which part of the gap comes from exact duplicate images, and which from photos of the same incident?

This measures **our** model under the two protocols. It does not re-measure the paper's model or labels.

## Dataset version check (Step 0)

Our downloaded copy has 31,795 images and 41,804 objects. Its per-class counts match the paper and the *Data in Brief* article exactly:

| Class | Images containing the class | Objects |
|---|---|---|
| barriers | 6,322 | 8,417 |
| sidewalks | 6,709 | 7,412 |
| pothole | 21,429 | 25,975 |
| **sum** | **34,460** | **41,804** |

So the paper's "34,460 images" is the **sum of per-class image counts**. 2,665 images contain two classes and are counted twice. The data is the same as in the paper, so no newer version needs to be downloaded.

The paper splits **per class** at random. An image that holds two classes is therefore listed under both classes, and could fall in train for one class and in test for the other. That would be a third leak path, on top of duplicates and shared incidents. Our random split assigns each image once, so it does not reproduce this effect, and our inflation estimate is a lower bound.

## Design

1. **Clean test C.** The v1 test split: 3,053 images, grouped by incident. C and every incident in it are held out of everything below.
2. **Random pool.** The remaining images, **keeping exact duplicates** (as the paper did), are split at random at the image level, 70/10/20, with seed 42. The splits are not stratified by incident.
3. **Model.** YOLO11s at 640 px, 50 epochs, the same settings as v1. Only the split changes.
4. **Test sets for that model:**
   - **R**: the random 20% test set;
   - **R-dedup**: R without images that are byte-identical to any train image;
   - **C**: the clean test set.
5. **Quantities:**
   - inflation = mAP(R) − mAP(C);
   - duplicate effect = mAP(R) − mAP(R-dedup);
   - incident effect = mAP(R-dedup) − mAP(C).

   Each is reported for mAP50 and mAP50-95, per class and overall.

## Statistics

- Metric code: `src/eval_ci.py`.
  - Predictions use conf 0.001, NMS IoU 0.7 and max 300 per image, as Ultralytics validation does.
  - AP uses the COCO 101-point interpolation.
- Each mAP has a 95% percentile bootstrap CI: 2,000 resamples, **resampling whole incidents**, seed 0.
- A difference counts as real only if its 95% CI excludes 0.
  - Same images, different models: paired bootstrap (identical resamples).
  - Different test sets (R vs C): independent bootstraps per set.
- Also reported, so the numbers line up with the paper's table: P, R and F1 at the confidence that maximizes F1 on the set being evaluated.

## What will not be done

- C is not used to choose anything.
- No re-running with other seeds to pick a favourable result. If time allows, extra seeds are reported as well, not instead.
