"""Audit a YOLO-format dataset before training.

Checks label integrity, class balance, objects per image, box sizes, EXIF
orientation, near-duplicate images and incident-level overlap between splits.

Usage:
    python src/audit_data.py --data "path/to/Visual Pollution Dataset" --out reports/data_audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["barriers", "sidewalks", "pothole"]
SPLITS = ["train", "val"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
EXIF_ORIENTATION = 0x0112
# Files are named "... INC-YY-MM-NNNNNN!timestamp": several photos of one incident ticket.
INCIDENT_RE = re.compile(r"(INC-\d{2}-\d{2}-\d+)")
# Box-size buckets by box area as a fraction of image area.
SIZE_BUCKETS = [("tiny <0.5%", 0.005), ("small 0.5-2%", 0.02), ("medium 2-10%", 0.10), ("large >=10%", 1.01)]
NEAR_DUP_BITS = 2  # dHash distance at or below this marks a near-duplicate *candidate*;
# low-texture (dark, blurry) photos can collide, so check candidates by eye.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="dataset root with images/{train,val} and labels/{train,val}")
    p.add_argument("--out", default=Path("reports/data_audit"), type=Path, help="output folder")
    p.add_argument("--workers", default=None, type=int, help="processes for image reading (default: CPU count)")
    p.add_argument("--limit", default=None, type=int, help="audit only the first N images per split (for a quick test)")
    return p.parse_args()


# ---------------------------------------------------------------- labels

def read_labels(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Return (boxes, problems) for one YOLO label file."""
    boxes, problems = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            problems.append(f"line {i}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            problems.append(f"line {i}: not numeric")
            continue
        if not 0 <= cls < len(CLASS_NAMES):
            problems.append(f"line {i}: class id {cls} out of range")
        if not all(0 <= v <= 1 for v in (cx, cy, w, h)) or w == 0 or h == 0:
            problems.append(f"line {i}: coordinates outside (0, 1]")
        boxes.append((cls, cx, cy, w, h))
    return boxes, problems


# ---------------------------------------------------------------- images

def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash, robust to resizing and recompression."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def image_info(path: Path) -> dict:
    try:
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        with Image.open(path) as img:
            orientation = img.getexif().get(EXIF_ORIENTATION, 1)
            return {"w": img.width, "h": img.height, "orientation": int(orientation), "hash": dhash(img), "md5": md5}
    except Exception as e:  # unreadable / truncated file
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- near-duplicates

def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1).reshape(x.shape)


def near_duplicates(query: np.ndarray, ref: np.ndarray, same_set: bool, chunk: int = 512) -> list[tuple[int, int, int]]:
    """Pairs (i, j, distance) with Hamming(query[i], ref[j]) <= NEAR_DUP_BITS."""
    pairs = []
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk, None]
        d = popcount64(np.bitwise_xor(q, ref[None, :]))
        for i, j in zip(*np.nonzero(d <= NEAR_DUP_BITS)):
            gi = start + i
            if same_set and j <= gi:
                continue
            pairs.append((int(gi), int(j), int(d[i, j])))
    return pairs


# ---------------------------------------------------------------- report

def bucket(area: float) -> str:
    return next(name for name, upper in SIZE_BUCKETS if area < upper)


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "n/a"


def plot(stats: dict, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(CLASS_NAMES))
    for k, split in enumerate(SPLITS):
        axes[0].bar(x + (k - 0.5) * 0.4, [stats[split]["objects_per_class"][c] for c in CLASS_NAMES], 0.4, label=split)
    axes[0].set(xticks=x, xticklabels=CLASS_NAMES, title="Objects per class")
    axes[0].legend()

    counts = Counter()
    for split in SPLITS:
        counts.update(stats[split]["objects_per_image_hist"])
    ks = sorted(int(k) for k in counts)
    axes[1].bar([str(k) for k in ks], [counts[k] for k in ks])
    axes[1].set(title="Objects per image (train+val)", xlabel="objects")

    for c in CLASS_NAMES:
        areas = np.array(stats["_areas"][c])
        if len(areas):
            axes[2].hist(np.log10(np.clip(areas, 1e-5, 1)), bins=40, histtype="step", label=c)
    axes[2].set(title="Box area / image area", xlabel="log10(area fraction)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=110)
    plt.close(fig)
    files.append("overview.png")
    return files


def write_report(stats: dict, exact: dict, dups: dict, incidents: dict, out: Path, figures: list[str]) -> None:
    L = ["# Data audit report", ""]
    L += ["## Files and label integrity", "", "| | train | val |", "|---|---|---|"]
    for key, label in [("images", "Images"), ("labels", "Label files"), ("missing_label", "Images without a label file"),
                       ("orphan_label", "Label files without an image"), ("empty_label", "Empty label files"),
                       ("bad_files", "Label files with problems"), ("unreadable", "Unreadable images")]:
        L.append(f"| {label} | {stats['train'][key]:,} | {stats['val'][key]:,} |")
    L.append("")

    L += ["## Class balance", "", "| Class | train objects | val objects | images containing it (train+val) |", "|---|---|---|---|"]
    for c in CLASS_NAMES:
        imgs = stats["train"]["images_with_class"][c] + stats["val"]["images_with_class"][c]
        L.append(f"| {c} | {stats['train']['objects_per_class'][c]:,} | {stats['val']['objects_per_class'][c]:,} | {imgs:,} |")
    tot = sum(stats[s]["objects"] for s in SPLITS)
    L += ["", f"Total objects: **{tot:,}** in {sum(stats[s]['labelled_images'] for s in SPLITS):,} labelled images.", ""]

    multi = sum(stats[s]["multi_object_images"] for s in SPLITS)
    dropped = sum(stats[s]["objects"] - stats[s]["labelled_images"] for s in SPLITS)
    L += ["## Objects per image", "",
          f"- Images with more than one object: **{multi:,}** ({pct(multi, sum(stats[s]['labelled_images'] for s in SPLITS))})",
          f"- Objects the v0 notebook discards by keeping only the first line: **{dropped:,}** ({pct(dropped, tot)} of all objects)", ""]

    L += ["## Box sizes (area as a fraction of the image)", "", "| Class | " + " | ".join(n for n, _ in SIZE_BUCKETS) + " | median |",
          "|---|" + "---|" * (len(SIZE_BUCKETS) + 1)]
    for c in CLASS_NAMES:
        b = Counter(bucket(a) for a in stats["_areas"][c])
        n = len(stats["_areas"][c])
        med = float(np.median(stats["_areas"][c])) if n else 0
        L.append(f"| {c} | " + " | ".join(pct(b[name], n) for name, _ in SIZE_BUCKETS) + f" | {100 * med:.2f}% |")
    L += ["", "At 128×128 input, a box under 0.5% of the image is smaller than about 9×9 pixels.", ""]

    L += ["## Image sizes and EXIF orientation", ""]
    for s in SPLITS:
        top = ", ".join(f"{k} ({v:,})" for k, v in stats[s]["sizes"].most_common(5))
        L.append(f"- **{s}** most common sizes: {top}")
    for s in SPLITS:
        o = stats[s]["orientation"]
        rotated = sum(v for k, v in o.items() if k != 1)
        L.append(f"- **{s}** EXIF orientation: " + ", ".join(f"{k}: {v:,}" for k, v in sorted(o.items()))
                 + f" → rotated/flipped flag on **{rotated:,}** images ({pct(rotated, stats[s]['images'] - stats[s]['unreadable'])})")
    L += ["", "Orientation 1 means no rotation. Any other value means the viewer rotates the image, so labels may not "
          "match what libraries that ignore EXIF (or that apply it) see. Check a sample by eye before training.", ""]

    L += ["## Exact duplicates (identical file bytes)", "",
          f"- **Val images that also appear in train: {exact['val_in_train']:,}** ({pct(exact['val_in_train'], stats['val']['images'])} of val). "
          "Listed in `exact_duplicates_val_train.txt`.",
          f"- Extra copies within train: {exact['train_within']:,}; within val: {exact['val_within']:,}",
          f"- Unique images across both splits: **{exact['unique_images']:,}** of {stats['train']['images'] + stats['val']['images']:,}", ""]

    L += ["## Near-duplicate candidates (dHash, Hamming distance ≤ %d)" % NEAR_DUP_BITS, "",
          "Includes the exact duplicates above. Dark or blurry photos can match by chance, so treat these as candidates "
          "and check a sample by eye.", "",
          f"- Within train: **{dups['train_train']:,}** pairs",
          f"- Within val: **{dups['val_val']:,}** pairs",
          f"- **Across train and val: {dups['val_train']:,} pairs**, involving {dups['val_images_with_train_dup']:,} val images "
          f"({pct(dups['val_images_with_train_dup'], stats['val']['images'])})", ""]

    L += ["## Incident tickets shared between splits", "",
          "File names contain an incident id (`INC-YY-MM-NNNNNN`); one incident usually has several photos of the same spot.", "",
          f"- Incidents in train: {incidents['train']:,}; in val: {incidents['val']:,}",
          f"- Images without an incident id: {incidents['no_id']:,}",
          f"- **Incidents present in both splits: {incidents['shared']:,}**, covering {incidents['val_images_shared']:,} val images "
          f"({pct(incidents['val_images_shared'], stats['val']['images'])})",
          f"- Photos per incident: median {incidents['median_photos']}, max {incidents['max_photos']}", ""]

    L += ["## Figures", ""] + [f"![{f}]({f})" for f in figures] + [""]
    if stats["problem_examples"]:
        L += ["## Label problem examples", ""] + [f"- `{e}`" for e in stats["problem_examples"][:20]] + [""]
    (out / "report.md").write_text("\n".join(L))


# ---------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"_areas": {c: [] for c in CLASS_NAMES}, "problem_examples": []}
    hashes, names, md5s = {}, {}, {}

    for split in SPLITS:
        img_dir, lbl_dir = args.data / "images" / split, args.data / "labels" / split
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if args.limit:
            images = images[:args.limit]
        label_stems = {p.stem for p in lbl_dir.glob("*.txt")}
        s = {"images": len(images), "labels": len(label_stems), "missing_label": 0, "empty_label": 0, "bad_files": 0,
             "orphan_label": 0 if args.limit else len(label_stems - {p.stem for p in images}),
             "unreadable": 0, "labelled_images": 0, "objects": 0, "multi_object_images": 0,
             "objects_per_class": Counter(), "images_with_class": Counter(), "objects_per_image_hist": Counter(),
             "sizes": Counter(), "orientation": Counter()}

        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            if img.stem not in label_stems:
                s["missing_label"] += 1
                continue
            boxes, problems = read_labels(lbl)
            if problems:
                s["bad_files"] += 1
                stats["problem_examples"] += [f"{split}/{lbl.name}: {p}" for p in problems[:2]]
            if not boxes:
                s["empty_label"] += 1
                continue
            s["labelled_images"] += 1
            s["objects"] += len(boxes)
            s["objects_per_image_hist"][len(boxes)] += 1
            s["multi_object_images"] += len(boxes) > 1
            for cls in {b[0] for b in boxes if 0 <= b[0] < len(CLASS_NAMES)}:
                s["images_with_class"][CLASS_NAMES[cls]] += 1
            for cls, _, _, w, h in boxes:
                if 0 <= cls < len(CLASS_NAMES):
                    s["objects_per_class"][CLASS_NAMES[cls]] += 1
                    stats["_areas"][CLASS_NAMES[cls]].append(w * h)

        print(f"[{split}] reading {len(images):,} images ...", flush=True)
        with ProcessPoolExecutor(args.workers) as ex:
            infos = list(ex.map(image_info, images, chunksize=64))
        ok_hashes, ok_names = [], []
        md5s[split] = {}
        for img, info in zip(images, infos):
            if "error" in info:
                s["unreadable"] += 1
                continue
            s["sizes"][f"{info['w']}x{info['h']}"] += 1
            s["orientation"][info["orientation"]] += 1
            ok_hashes.append(info["hash"])
            md5s[split].setdefault(info["md5"], []).append(img.name)
            ok_names.append(img.name)
        hashes[split] = np.array(ok_hashes, dtype=np.uint64)
        names[split] = ok_names
        stats[split] = s

    shared_md5 = set(md5s["train"]) & set(md5s["val"])
    exact = {
        "train_within": sum(len(v) - 1 for v in md5s["train"].values()),
        "val_within": sum(len(v) - 1 for v in md5s["val"].values()),
        "val_in_train": sum(len(md5s["val"][m]) for m in shared_md5),
        "unique_images": len(set(md5s["train"]) | set(md5s["val"])),
    }
    (args.out / "exact_duplicates_val_train.txt").write_text(
        "".join(n + "\n" for m in sorted(shared_md5) for n in md5s["val"][m]))

    print("finding near-duplicates ...", flush=True)
    tt = near_duplicates(hashes["train"], hashes["train"], same_set=True)
    vv = near_duplicates(hashes["val"], hashes["val"], same_set=True)
    vt = near_duplicates(hashes["val"], hashes["train"], same_set=False)
    dups = {"train_train": len(tt), "val_val": len(vv), "val_train": len(vt),
            "val_images_with_train_dup": len({i for i, _, _ in vt})}
    (args.out / "near_duplicates_val_train.csv").write_text(
        "val_image,train_image,hamming\n" + "".join(f'"{names["val"][i]}","{names["train"][j]}",{d}\n' for i, j, d in vt))

    by_split: dict[str, Counter] = {}
    no_id = 0
    for split in SPLITS:
        ids = []
        for n in names[split]:
            m = INCIDENT_RE.search(n)
            if m:
                ids.append(m.group(1))
            else:
                no_id += 1
        by_split[split] = Counter(ids)
    shared = set(by_split["train"]) & set(by_split["val"])
    per_incident = by_split["train"] + by_split["val"]
    incidents = {"train": len(by_split["train"]), "val": len(by_split["val"]), "no_id": no_id, "shared": len(shared),
                 "val_images_shared": sum(by_split["val"][i] for i in shared),
                 "median_photos": int(np.median(list(per_incident.values()))) if per_incident else 0,
                 "max_photos": max(per_incident.values(), default=0)}

    figures = plot(stats, args.out)
    write_report(stats, exact, dups, incidents, args.out, figures)

    summary = {s: {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats[s].items()} for s in SPLITS}
    for s in SPLITS:
        summary[s]["orientation"] = {str(k): v for k, v in summary[s]["orientation"].items()}
        summary[s]["objects_per_image_hist"] = {str(k): v for k, v in summary[s]["objects_per_image_hist"].items()}
    (args.out / "summary.json").write_text(json.dumps({"splits": summary, "exact_duplicates": exact, "near_duplicates": dups, "incidents": incidents}, indent=2))
    print(f"report written to {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
