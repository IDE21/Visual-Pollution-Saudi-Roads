"""Sliced inference (SAHI) for a YOLO model, saved in the eval_ci.py npz format (scored with `eval_ci.py score`).

Each image is cut into overlapping slices; every slice is predicted at the model's input size,
and the slice boxes are merged with the full-image prediction. This helps small objects.

    python src/predict_sahi.py --weights best.pt --list list.txt --out preds.npz --slice 320
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval_ci import load_gt, match, read_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--slice", type=int, default=320, help="slice side in pixels")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    from PIL import Image
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=a.weights, confidence_threshold=a.conf, device=a.device, image_size=a.imgsz
    )
    imgs = read_list(a.list)
    names, img_idx, conf, cls, tp, gt_img, gt_cls = [], [], [], [], [], [], []
    for k, p in enumerate(imgs):
        W, H = Image.open(p).size
        res = get_sliced_prediction(
            str(p), model, slice_height=a.slice, slice_width=a.slice,
            overlap_height_ratio=a.overlap, overlap_width_ratio=a.overlap,
            perform_standard_pred=True, postprocess_type="NMS", postprocess_match_threshold=0.5, verbose=0,
        )
        ob = res.object_prediction_list
        p_cls = np.array([o.category.id for o in ob], int)
        p_conf = np.array([o.score.value for o in ob], float)
        p_box = np.array([o.bbox.to_xyxy() for o in ob], float).reshape(-1, 4) / [W, H, W, H]
        g_cls, g_box = load_gt(p)
        names.append(p.name)
        img_idx += [k] * len(p_cls)
        conf.append(p_conf)
        cls.append(p_cls)
        tp.append(match(p_cls, p_box, p_conf, g_cls, g_box))
        gt_img += [k] * len(g_cls)
        gt_cls.append(g_cls)
        if (k + 1) % 200 == 0 or k + 1 == len(imgs):
            print(f"{k + 1}/{len(imgs)}", flush=True)
    np.savez_compressed(
        a.out, names=np.array(names), pred_img=np.array(img_idx, int), conf=np.concatenate(conf),
        cls=np.concatenate(cls), tp=np.concatenate(tp), gt_img=np.array(gt_img, int), gt_cls=np.concatenate(gt_cls),
    )
    print(f"saved {a.out}: {len(names)} images")

if __name__ == "__main__":
    main()
