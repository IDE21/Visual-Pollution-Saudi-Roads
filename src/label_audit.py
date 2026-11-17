"""Semi-automatic label audit (plan v2, step 4).

Compares a model's predictions on a held-out split (val) with the released labels and flags
likely label errors. Only train/val are ever audited; the clean test set C is never touched.

    predict  run the model, save every box (conf >= 0.05) to an npz
    flag     match predictions to labels and write a ranked candidates.csv

Issue types (IoU = box overlap, conf = model confidence):
    missing_label  conf >= 0.5 and no label of any class with IoU >= 0.3
    wrong_class    conf >= 0.5, a label of ANOTHER class with IoU >= 0.5, and no same-class label with IoU >= 0.3
    bad_box        conf >= 0.5, best same-class label IoU in [0.1, 0.5) and that label matches nothing better
    spurious_label a label with no prediction of its class at IoU >= 0.1 and conf >= 0.1

The score used for ranking is the model confidence (spurious_label: 1 - best conf near it).
These are candidates, not verdicts; a visual check decides.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval_ci import box_iou, load_gt, read_list

NAMES = ["barriers", "sidewalks", "pothole"]


def cmd_predict(a) -> None:
    from ultralytics import YOLO

    model = YOLO(a.weights)
    imgs = read_list(a.list)
    rows = []
    for s in range(0, len(imgs), a.batch):
        chunk = [str(p) for p in imgs[s : s + a.batch]]
        for k, r in enumerate(model.predict(chunk, imgsz=a.imgsz, conf=0.05, device=a.device, verbose=False)):
            b = r.boxes
            for c, conf, box in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xyxyn.cpu().numpy()):
                rows.append([s + k, int(c), float(conf), *box])
        print(f"{min(s + a.batch, len(imgs))}/{len(imgs)}", flush=True)
    np.savez_compressed(a.out, paths=np.array([str(p) for p in imgs]), preds=np.array(rows, float).reshape(-1, 7))


def cmd_flag(a) -> None:
    d = np.load(a.preds)
    paths, P = d["paths"], d["preds"]
    out = []
    n_gt = 0
    for i, path in enumerate(paths):
        g_cls, g_box = load_gt(Path(path))
        n_gt += len(g_cls)
        p = P[P[:, 0] == i]
        p_cls, p_conf, p_box = p[:, 1].astype(int), p[:, 2], p[:, 3:7]
        iou = box_iou(p_box, g_box) if len(p) and len(g_cls) else np.zeros((len(p), len(g_cls)))
        same = p_cls[:, None] == g_cls[None, :] if len(g_cls) else np.zeros((len(p), 0), bool)
        best_same_for_gt = np.where(same, iou, 0).max(0) if len(p) else np.zeros(len(g_cls))

        for j in np.where(p_conf >= 0.5)[0]:
            any_iou = iou[j].max() if len(g_cls) else 0.0
            s_iou = np.where(same[j], iou[j], 0)
            o_iou = np.where(~same[j], iou[j], 0)
            rec = dict(image=path, cls=NAMES[p_cls[j]], score=round(float(p_conf[j]), 3), box=" ".join(f"{v:.4f}" for v in p_box[j]))
            if any_iou < 0.3:
                out.append(dict(rec, issue="missing_label", label_cls="", label_box=""))
            elif s_iou.max(initial=0) < 0.3 and o_iou.max(initial=0) >= 0.5:
                g = int(o_iou.argmax())
                out.append(dict(rec, issue="wrong_class", label_cls=NAMES[g_cls[g]], label_box=" ".join(f"{v:.4f}" for v in g_box[g])))
            elif 0.1 <= s_iou.max(initial=0) < 0.5:
                g = int(s_iou.argmax())
                if best_same_for_gt[g] < 0.5:  # that label is not well matched by any other prediction
                    out.append(dict(rec, issue="bad_box", label_cls=NAMES[g_cls[g]], label_box=" ".join(f"{v:.4f}" for v in g_box[g])))

        for g in range(len(g_cls)):
            near = same[:, g] & (iou[:, g] >= 0.1) if len(p) else np.zeros(0, bool)
            best = p_conf[near].max() if near.any() else 0.0
            if best < 0.1:
                out.append(dict(image=path, cls=NAMES[g_cls[g]], score=round(1 - float(best), 3), box="", issue="spurious_label",
                                label_cls=NAMES[g_cls[g]], label_box=" ".join(f"{v:.4f}" for v in g_box[g])))

    out.sort(key=lambda r: -r["score"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["issue", "cls", "label_cls", "score", "image", "box", "label_box"])
        w.writeheader()
        w.writerows(out)
    from collections import Counter

    c = Counter(r["issue"] for r in out)
    print(f"{len(paths)} images, {n_gt} labels, {len(out)} candidates: {dict(c)}")
    for issue in c:
        print(issue, dict(Counter(r["cls"] for r in out if r["issue"] == issue)))


PER_TYPE = {"missing_label": 35, "spurious_label": 35, "wrong_class": 15, "bad_box": 15}


def cmd_sample(a) -> None:
    """Stratified random sample of candidates, one crop per case.

    Crops show the released label in green and the model's box in red, with context around both.
    """
    from PIL import Image, ImageDraw

    rows = list(csv.DictReader(a.candidates.open()))
    rng = np.random.default_rng(a.seed)
    picked = []
    for issue, k in PER_TYPE.items():
        pool = [r for r in rows if r["issue"] == issue]
        picked += [pool[i] for i in sorted(rng.choice(len(pool), min(k, len(pool)), replace=False))]
    order = rng.permutation(len(picked))  # shuffle so the issue type is not obvious from the position
    picked = [picked[i] for i in order]
    a.out.mkdir(parents=True, exist_ok=True)
    for n, r in enumerate(picked, 1):
        im = Image.open(r["image"]).convert("RGB")
        W, H = im.size
        boxes = []
        for key, colour in (("label_box", (0, 220, 0)), ("box", (255, 40, 40))):
            if r[key]:
                x1, y1, x2, y2 = (float(v) for v in r[key].split())
                boxes.append(((x1 * W, y1 * H, x2 * W, y2 * H), colour))
        dr = ImageDraw.Draw(im)
        lw = max(3, W // 250)
        for b, colour in boxes:
            dr.rectangle(b, outline=colour, width=lw)
        xs = [v for b, _ in boxes for v in (b[0], b[2])]
        ys = [v for b, _ in boxes for v in (b[1], b[3])]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        side = max(max(xs) - min(xs), max(ys) - min(ys)) * 2.2 + 0.15 * max(W, H)
        crop = im.crop((max(0, cx - side / 2), max(0, cy - side / 2), min(W, cx + side / 2), min(H, cy + side / 2)))
        crop.thumbnail((640, 640))
        crop.save(a.out / f"{n:03d}.jpg", quality=88)
        r["case"] = n
    with (a.out / "sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "issue", "cls", "label_cls", "score", "image", "box", "label_box"])
        w.writeheader()
        w.writerows(picked)
    print(f"{len(picked)} cases -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--weights", required=True)
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="mps")
    f = sub.add_parser("flag")
    f.add_argument("preds", type=Path)
    f.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("sample")
    s.add_argument("candidates", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"predict": cmd_predict, "flag": cmd_flag, "sample": cmd_sample}[a.cmd](a)


if __name__ == "__main__":
    main()
