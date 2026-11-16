"""Detection metrics with bootstrap confidence intervals, resampled by incident.

Step 1 (predict, needs ultralytics): run a model on an image list and cache
per-image matches.
    python src/eval_ci.py predict --weights best.pt --list "<root>/test_v1.txt" --out preds_test.npz

Step 2 (score, numpy only): mAP50 / mAP50-95 per class with 95% CIs.
    python src/eval_ci.py score preds_test.npz [--paired other.npz] [--subset keep.txt] --json out.json

--paired compares two models on the same images: both are resampled with the
same incidents, which gives a CI for the difference.
--subset keeps only the images listed in a file (for example R-dedup).

The bootstrap resamples whole incident tickets (INC-YY-MM-NNNNNN in the file
name), because photos of one incident are not independent.
AP follows the Ultralytics/COCO 101-point interpolation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
INCIDENT_RE = re.compile(r"(INC-\d{2}-\d{2}-\d+)")
NAMES = ["barriers", "sidewalks", "pothole"]


# ---------------------------------------------------------------- predict

def read_list(path: Path) -> list[Path]:
    root = path.parent
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            p = Path(line)
            out.append(p if p.is_absolute() else (root / p).resolve())
    return out


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_gt(img: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [l.split() for l in label_path(img).read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows, float)
    xywh = a[:, 1:5]
    xyxy = np.c_[xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2]
    return a[:, 0].astype(int), xyxy


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(pcls, pbox, pconf, gcls, gbox) -> np.ndarray:
    """COCO-style greedy matching by confidence. Returns tp[n_pred, 10]."""
    tp = np.zeros((len(pcls), len(IOU_THRESHOLDS)), bool)
    if len(pcls) == 0 or len(gcls) == 0:
        return tp
    iou = box_iou(pbox, gbox)
    iou[pcls[:, None] != gcls[None, :]] = 0
    order = np.argsort(-pconf)
    for t, thr in enumerate(IOU_THRESHOLDS):
        used = np.zeros(len(gcls), bool)
        for i in order:
            cand = np.where(~used & (iou[i] >= thr))[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                used[j] = True
                tp[i, t] = True
    return tp


def cmd_predict(args) -> None:
    from ultralytics import YOLO

    imgs = read_list(args.list)
    model = YOLO(args.weights)
    names, img_idx, conf, cls, tp, gt_img, gt_cls = [], [], [], [], [], [], []
    def results():
        for s in range(0, len(imgs), args.batch):  # a list source is one batch in Ultralytics, so chunk it
            yield from model.predict(source=[str(p) for p in imgs[s:s + args.batch]], imgsz=args.imgsz, conf=0.001,
                                     iou=0.7, max_det=300, device=args.device, verbose=False)

    for k, r in enumerate(results()):
        img = imgs[k]
        g_cls, g_box = load_gt(img)
        b = r.boxes
        p_cls = b.cls.cpu().numpy().astype(int)
        p_conf = b.conf.cpu().numpy()
        p_box = b.xyxyn.cpu().numpy()
        t = match(p_cls, p_box, p_conf, g_cls, g_box)
        names.append(img.name)
        img_idx += [k] * len(p_cls)
        conf.append(p_conf)
        cls.append(p_cls)
        tp.append(t)
        gt_img += [k] * len(g_cls)
        gt_cls.append(g_cls)
        if (k + 1) % 500 == 0:
            print(f"{k + 1}/{len(imgs)}", flush=True)
    np.savez_compressed(
        args.out, names=np.array(names), pred_img=np.array(img_idx, int), conf=np.concatenate(conf),
        cls=np.concatenate(cls), tp=np.concatenate(tp), gt_img=np.array(gt_img, int), gt_cls=np.concatenate(gt_cls),
    )
    print(f"saved {args.out}: {len(names)} images")


# ---------------------------------------------------------------- score

def compute_ap(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> np.ndarray:
    """AP per IoU threshold (10,) for one class."""
    if n_gt == 0:
        return np.full(len(IOU_THRESHOLDS), np.nan)
    if len(conf) == 0:
        return np.zeros(len(IOU_THRESHOLDS))
    o = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[o], axis=0)
    fpc = np.cumsum(~tp[o], axis=0)
    recall = tpc / n_gt
    precision = tpc / (tpc + fpc)
    x = np.linspace(0, 1, 101)
    ap = np.empty(tp.shape[1])
    for t in range(tp.shape[1]):
        mrec = np.r_[0.0, recall[:, t], 1.0]
        mpre = np.r_[1.0, precision[:, t], 0.0]
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        ap[t] = np.trapezoid(np.interp(x, mrec, mpre), x)
    return ap


def best_f1(tp50: np.ndarray, conf: np.ndarray, n_gt: int) -> dict:
    if n_gt == 0 or len(conf) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "conf": None}
    o = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp50[o])
    p = tpc / np.arange(1, len(o) + 1)
    r = tpc / n_gt
    f1 = 2 * p * r / (p + r + 1e-9)
    i = int(np.argmax(f1))
    return {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "conf": float(conf[o][i])}


class Preds:
    def __init__(self, path: Path, keep: set[str] | None = None):
        d = np.load(path)
        names = d["names"]
        sel = np.ones(len(names), bool) if keep is None else np.array([n in keep for n in names])
        self.names = names[sel]
        old_to_new = -np.ones(len(names), int)
        old_to_new[np.where(sel)[0]] = np.arange(sel.sum())
        pm = sel[d["pred_img"]]
        gm = sel[d["gt_img"]]
        self.pred_img = old_to_new[d["pred_img"][pm]]
        self.conf, self.cls, self.tp = d["conf"][pm], d["cls"][pm], d["tp"][pm]
        self.gt_img, self.gt_cls = old_to_new[d["gt_img"][gm]], d["gt_cls"][gm]
        n = len(self.names)
        # per image, per class: index ranges into sorted prediction arrays and GT counts
        self.n_gt = np.zeros((n, len(NAMES)), int)
        np.add.at(self.n_gt, (self.gt_img, self.gt_cls), 1)
        self.by_class = []
        for c in range(len(NAMES)):
            m = self.cls == c
            img, conf, tp = self.pred_img[m], self.conf[m], self.tp[m]
            o = np.argsort(img, kind="stable")
            img, conf, tp = img[o], conf[o], tp[o]
            starts = np.searchsorted(img, np.arange(n))
            ends = np.searchsorted(img, np.arange(n), side="right")
            self.by_class.append((conf, tp, starts, ends))

    def metrics(self, weights: np.ndarray) -> np.ndarray:
        """weights[n_images] = how many times each image is drawn. Returns ap[class, 10]."""
        out = np.empty((len(NAMES), len(IOU_THRESHOLDS)))
        for c, (conf, tp, starts, ends) in enumerate(self.by_class):
            counts = weights[np.repeat(np.arange(len(weights)), ends - starts)]
            out[c] = compute_ap(np.repeat(tp, counts, axis=0), np.repeat(conf, counts), int((self.n_gt[:, c] * weights).sum()))
        return out

    def f1_table(self) -> dict:
        res = {}
        for c, (conf, tp, _, _) in enumerate(self.by_class):
            res[NAMES[c]] = best_f1(tp[:, 0], conf, int(self.n_gt[:, c].sum()))
        return res


def incidents(names) -> np.ndarray:
    keys = [(m.group(1) if (m := INCIDENT_RE.search(str(n))) else str(n)) for n in names]
    _, inv = np.unique(keys, return_inverse=True)
    return inv


def summarize(ap: np.ndarray) -> dict:
    """ap[class, 10] -> mAP50 / mAP50-95 per class and overall."""
    d = {NAMES[c]: {"mAP50": ap[c, 0], "mAP50-95": ap[c].mean()} for c in range(len(NAMES))}
    d["all"] = {"mAP50": np.nanmean(ap[:, 0]), "mAP50-95": np.nanmean(ap.mean(1))}
    return d


def flat(d: dict) -> dict:
    return {(k, m): float(v) for k, sub in d.items() for m, v in sub.items()}


def cmd_score(args) -> None:
    keep = None
    if args.subset:
        keep = {Path(l.strip()).name for l in args.subset.read_text().splitlines() if l.strip()}
    a = Preds(args.preds, keep)
    b = None
    if args.paired:
        b = Preds(args.paired, keep)
        if list(a.names) != list(b.names):
            raise SystemExit("--paired files must cover the same images in the same order")
    inc = incidents(a.names)
    n_inc = inc.max() + 1
    rng = np.random.default_rng(args.seed)

    point_a = flat(summarize(a.metrics(np.ones(len(a.names), int))))
    point_b = flat(summarize(b.metrics(np.ones(len(b.names), int)))) if b else None
    boots_a, boots_d = [], []
    for i in range(args.n_boot):
        draw = np.bincount(rng.integers(0, n_inc, n_inc), minlength=n_inc)
        w = draw[inc]
        fa = flat(summarize(a.metrics(w)))
        boots_a.append(fa)
        if b:
            fb = flat(summarize(b.metrics(w)))
            boots_d.append({k: fa[k] - fb[k] for k in fa})
        if (i + 1) % 500 == 0:
            print(f"bootstrap {i + 1}/{args.n_boot}", flush=True)

    def ci(boots, k):
        v = np.array([x[k] for x in boots])
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    report = {"preds": str(args.preds), "images": len(a.names), "incidents": int(n_inc), "n_boot": args.n_boot,
              "seed": args.seed, "metrics": {}, "best_f1_at_iou50": a.f1_table()}
    for k, v in point_a.items():
        report["metrics"].setdefault(k[0], {})[k[1]] = {"value": v, "ci95": ci(boots_a, k)}
    if b:
        report["paired_with"] = str(args.paired)
        report["difference_a_minus_b"] = {}
        for k in point_a:
            report["difference_a_minus_b"].setdefault(k[0], {})[k[1]] = {
                "value": point_a[k] - point_b[k], "ci95": ci(boots_d, k)}

    print(f"\n{len(a.names)} images, {n_inc} incidents, {args.n_boot} bootstrap resamples (by incident)")
    print(f"{'class':<10} {'mAP50':>22} {'mAP50-95':>22}")
    for cl, m in report["metrics"].items():
        row = [f"{m[x]['value']:.3f} [{m[x]['ci95'][0]:.3f}, {m[x]['ci95'][1]:.3f}]" for x in ("mAP50", "mAP50-95")]
        print(f"{cl:<10} {row[0]:>22} {row[1]:>22}")
    if b:
        print("\ndifference (this − paired):")
        for cl, m in report["difference_a_minus_b"].items():
            row = [f"{m[x]['value']:+.3f} [{m[x]['ci95'][0]:+.3f}, {m[x]['ci95'][1]:+.3f}]" for x in ("mAP50", "mAP50-95")]
            print(f"{cl:<10} {row[0]:>22} {row[1]:>22}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwritten {args.json}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("predict")
    pp.add_argument("--weights", required=True)
    pp.add_argument("--list", required=True, type=Path, help="image list file (paths relative to its folder)")
    pp.add_argument("--out", required=True, type=Path)
    pp.add_argument("--imgsz", default=640, type=int)
    pp.add_argument("--batch", default=16, type=int)
    pp.add_argument("--device", default=None)
    ps = sub.add_parser("score")
    ps.add_argument("preds", type=Path)
    ps.add_argument("--paired", type=Path)
    ps.add_argument("--subset", type=Path, help="keep only images whose file names are listed here")
    ps.add_argument("--n-boot", default=2000, type=int)
    ps.add_argument("--seed", default=0, type=int)
    ps.add_argument("--json", type=Path)
    args = p.parse_args()
    cmd_predict(args) if args.cmd == "predict" else cmd_score(args)


if __name__ == "__main__":
    main()
