# Visual-Pollution-Saudi-Roads

Detecting **barriers**, **dilapidated sidewalks** and **potholes** in photos of Saudi public roads. The project also re-evaluates the dataset paper's model on a leak-free test set.

> **Status:** v1 released; plan v2 in progress ([`docs/PLAN_v2.md`](docs/PLAN_v2.md)). Weights and a demo are not published yet.

## Key results

| | Result |
|---|---|
| **v1** (YOLO11s, 640 px) on a clean test set, split by incident | **mAP50 0.699 [0.680, 0.716], mAP50-95 0.473** |
| Dataset paper (YOLOv5x, random split) | 93% mAP (type not stated) |
| Effect of the paper's random split on our model | Only **+2.4 mAP50** [0.1, 4.8], so leakage does not explain the gap |
| Same external pothole set as the paper (1,482 images, no fine-tuning) | **0.551** vs the paper's 70%: **the paper is ahead** |
| Label errors among the model's flagged cases on val | **56%** [42, 69], about 11–18% of all val labels |
| 960 px vs 640 px (paired, val) | No difference: mAP50-95 **+0.000** [−0.010, +0.009], at 2× the cost |

All CIs are 95% bootstrap intervals, resampled by incident (by image for the external set).

## v1 on the clean test set

3,053 images, never used in training or model selection.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| barriers | 0.627 | 0.637 | 0.657 | 0.491 |
| sidewalks | 0.604 | 0.619 | 0.643 | 0.382 |
| pothole | 0.714 | 0.734 | 0.797 | 0.546 |
| **all** | **0.649** | **0.663** | **0.699** | **0.473** |

- **Setup:** COCO-pretrained YOLO11s, 640 px, 50 epochs, batch 16, seed 42, default augmentation. Best epoch 46; val mAP was still rising.
- **Errors:** when an object is found, its class is almost always right (confusion-matrix diagonal 0.93–0.98). Most errors are false alarms on background, and 55% of these are predicted as potholes.
- **Speed:** about 5 ms per image on an Apple M5 Pro (batch 16, NMS included).
- Outputs, curves and plots: [`reports/yolo11s_640_v1`](reports/yolo11s_640_v1).

![Test predictions](reports/yolo11s_640_v1/test/val_batch0_pred.jpg)

## What we found

1. **The official split leaks.** 16% of the val images are byte-identical copies of train images, and 1,648 incidents have photos in both splits. v1 removes 1,271 duplicates and splits the data 80/10/10 by incident. → [data audit](reports/data_audit/report.md), [`splits/`](splits/summary.md)
2. **Leakage is small for our model.** We trained YOLO11s on a random 70/10/20 split, as in the paper, and it scored only 2.4 mAP50 points higher. The rest of the gap to 93% may come from the much larger YOLOv5x, the per-class split, the active-learning labels, or the mAP definition. The paper does not give enough detail to tell. → [Track A](reports/trackA/report.md)
3. **The paper's model is better on the external potholes**, 70% vs our 55%. External potholes are about 5× smaller and 3× denser than ours. Larger inference sizes and SAHI slicing made v1 *worse* (−3 to −11 points), and training at 960 px helped by only one point. The gap is mostly a difference in what counts as a pothole, not in resolution. → [external benchmark](reports/external_pothole/report.md)
4. **The labels are noisy.** The model flagged 1,046 likely label errors on val. The owner reviewed a random 50 of them blind, and 56% were real label errors: mostly missing or false labels, rarely loose boxes. Our measured scores are therefore pessimistic, and the class definitions need a written guide. → [label audit](reports/label_audit/report.md), [draft guide](docs/LABEL_GUIDE.md)
5. **Resolution is not the bottleneck.** We compared 640 and 960 px under the same settings, and the overall difference was zero. Potholes gained +0.9 mAP50-95, which is real but small. We keep 640 px. → [resolution screening](reports/resolution_screen/report.md)

**Limits:**
- One seed per setting.
- The label audit covers 50 human-judged cases from one reviewer.
- The paper's external-test protocol is not documented, so that comparison is approximate.

## Next steps

From [plan v2](docs/PLAN_v2.md) and the [label-cleaning plan](docs/PLAN_labels.md):
1. A baseline run at 640 px on all data for 100 epochs.
2. Label cleaning, starting with a cheap 2-fold pilot and then human review.
3. A single final evaluation on the clean test set and the external set.
4. Publishing the weights with a model card, and a Gradio demo.

## Reproduce

```bash
pip install -r requirements-train.txt
python src/make_splits.py --data "<dataset root>" --install
python src/train.py --data "<dataset root>/data_v1.yaml"
python src/eval_ci.py predict --weights runs/.../best.pt --list "<dataset root>/test_v1.txt" --out preds.npz
python src/eval_ci.py score preds.npz
```

Other scripts:

| Script | Purpose |
|---|---|
| [`scripts/screen_resolution.sh`](scripts/screen_resolution.sh) | 640 vs 960 screening |
| [`src/prep_external_pothole.py`](src/prep_external_pothole.py) | External pothole set |
| [`src/predict_sahi.py`](src/predict_sahi.py) | SAHI inference |
| [`src/label_audit.py`](src/label_audit.py) | Label audit |

## Dataset

**Saudi Arabia Public Roads Visual Pollution Dataset**, AlElaiwi et al. (2022), on Mendeley Data under **CC BY 4.0**.
- 31,795 images and 41,804 labelled objects.
- Classes used here: `barriers`, `sidewalks`, `pothole`.
- The labels are the dataset authors' work, used as published.

```
AlElaiwi, M., Ahmad, H., Hussain, J., Al-antari, M., Almarri, B., & Azhar, A. (2022).
Saudi Arabia Public Roads Visual Pollution Dataset (Version 1) [Data set]. Mendeley Data.
https://doi.org/10.17632/bb7b8vtwry.1
```

The dataset paper is AlElaiwi et al. (2022), [*Mathematics* 11(1):186](https://doi.org/10.3390/math11010186). The external pothole set is the Roboflow [Pothole Detection Dataset v2](https://universe.roboflow.com/aegis/pothole-detection-i00zy), CC BY 4.0.

## History

**v0** was a small custom CNN that predicted one box per image. Its evaluation leaked, and it reached an accuracy of 0.720 and a mean IoU of 0.166. Its corrected results and known issues are in [`docs/v0.md`](docs/v0.md).

## License

- **Code:** [MIT](LICENSE).
- **Images and labels:** they follow the original dataset's licence.
- **Model weights:** trained with Ultralytics, so they are subject to its AGPL-3.0 licence.
