"""Build leak-free train/val/test splits for the visual pollution dataset.

The official split has byte-identical images in both train and val, and
photos of the same incident ticket on both sides. This script:

1. pools the official train and val images,
2. removes exact duplicates (same file bytes), keeping one copy,
3. groups images by incident ticket (INC-YY-MM-NNNNNN in the file name),
4. assigns whole groups to train/val/test so that each split gets close to
   the target share of every class, with a fixed seed.

Outputs (committed to the repo):
    splits/{train,val,test}.txt   one image per line, relative to the dataset root
    splits/summary.md             counts and checks

With --install, it also writes Ultralytics-ready lists and data.yaml into the
dataset root, so training can use:  yolo train data="<root>/data_v1.yaml" ...

Usage:
    python src/make_splits.py --data "path/to/Visual Pollution Dataset" [--install]
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CLASS_NAMES = ["barriers", "sidewalks", "pothole"]
SOURCE_SPLITS = ["train", "val"]
OUT_SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
INCIDENT_RE = re.compile(r"(INC-\d{2}-\d{2}-\d+)")
VERSION = "v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="dataset root with images/{train,val} and labels/{train,val}")
    p.add_argument("--out", default=Path("splits"), type=Path, help="where to write the split lists")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--install", action="store_true", help="also write Ultralytics lists and data yaml into the dataset root")
    return p.parse_args()


def label_counts(path: Path) -> Counter:
    c = Counter()
    for line in path.read_text().splitlines():
        if line.strip():
            c[int(line.split()[0])] += 1
    return c


def main() -> None:
    args = parse_args()
    root = args.data

    # 1. pool images and dedupe by content
    by_md5: dict[str, list[str]] = defaultdict(list)
    for split in SOURCE_SPLITS:
        for img in sorted((root / "images" / split).glob("*.jpg")):
            rel = f"images/{split}/{img.name}"
            by_md5[hashlib.md5(img.read_bytes()).hexdigest()].append(rel)

    def label_of(rel: str) -> Path:
        return (root / rel.replace("images/", "labels/", 1)).with_suffix(".txt")

    kept: list[str] = []
    removed = 0
    label_conflicts: list[list[str]] = []
    for rels in by_md5.values():
        rels = sorted(rels)  # deterministic: prefer images/train/... over images/val/...
        if len({label_of(r).read_text().strip() for r in rels}) > 1:
            label_conflicts.append(rels)
        kept.append(rels[0])
        removed += len(rels) - 1

    # 2. group by incident ticket
    groups: dict[str, list[str]] = defaultdict(list)
    for rel in kept:
        m = INCIDENT_RE.search(rel)
        groups[m.group(1) if m else f"noid:{rel}"].append(rel)

    counts = {rel: label_counts(label_of(rel)) for rel in kept}
    group_vec = {g: np.array([sum(counts[r][k] for r in rels) for k in range(len(CLASS_NAMES))] + [len(rels)], float)
                 for g, rels in groups.items()}

    # 3. greedy stratified assignment of whole groups
    names = list(OUT_SPLITS)
    shares = np.array([OUT_SPLITS[s] for s in names])
    total = sum(group_vec.values())
    target = shares[:, None] * total[None, :]          # split x (classes + images)
    current = np.zeros_like(target)
    order = list(groups)
    random.Random(args.seed).shuffle(order)
    order.sort(key=lambda g: -group_vec[g][-1])        # big groups first, random order among equals
    assignment: dict[str, str] = {}
    for g in order:
        v = group_vec[g]
        # pick the split whose relative shortfall is largest, weighted by what this group contains
        deficit = (target - current) / np.maximum(target, 1)
        score = (deficit * (v > 0)).sum(1)
        i = int(np.argmax(score))
        current[i] += v
        assignment[g] = names[i]

    split_imgs: dict[str, list[str]] = {s: [] for s in names}
    for g, rels in groups.items():
        split_imgs[assignment[g]].extend(rels)

    # 4. checks
    md5_of = {r: m for m, rels in by_md5.items() for r in rels}
    for a in names:
        for b in names:
            if a < b:
                assert not {INCIDENT_RE.search(r).group(1) for r in split_imgs[a] if INCIDENT_RE.search(r)} & \
                       {INCIDENT_RE.search(r).group(1) for r in split_imgs[b] if INCIDENT_RE.search(r)}, f"incident leak {a}/{b}"
                assert not {md5_of[r] for r in split_imgs[a]} & {md5_of[r] for r in split_imgs[b]}, f"duplicate leak {a}/{b}"

    # 5. write
    args.out.mkdir(parents=True, exist_ok=True)
    for s in names:
        (args.out / f"{s}.txt").write_text("".join(f"{r}\n" for r in sorted(split_imgs[s])))

    L = [f"# Dataset splits ({VERSION})", "",
         f"Built by `src/make_splits.py` (seed {args.seed}) from the official train and val folders of the "
         "Saudi Arabia Public Roads Visual Pollution Dataset.", "",
         "## How they were built", "",
         f"- Images pooled: {sum(len(v) for v in by_md5.values()):,}",
         f"- Exact duplicates removed (identical bytes): {removed:,}",
         f"- Duplicates whose label files differ: {len(label_conflicts)} (first copy kept)",
         f"- Unique images: {len(kept):,} in {len(groups):,} incident groups",
         "- Whole incident groups are assigned to one split, so no incident and no identical image appears in two splits "
         "(checked by the script).", "",
         "## Result", "",
         "| Split | Images | Incidents | " + " | ".join(CLASS_NAMES) + " | objects |",
         "|---|---|---|" + "---|" * (len(CLASS_NAMES) + 1)]
    for i, s in enumerate(names):
        objs = current[i][:-1]
        n_groups = sum(1 for g in assignment if assignment[g] == s)
        cells = " | ".join(f"{int(objs[k]):,} ({100 * objs[k] / total[k]:.1f}%)" for k in range(len(CLASS_NAMES)))
        L.append(f"| {s} | {len(split_imgs[s]):,} ({100 * len(split_imgs[s]) / len(kept):.1f}%) | {n_groups:,} | {cells} | {int(objs.sum()):,} |")
    L += ["", "Percentages are the share of each class's objects that landed in that split (target: "
          + ", ".join(f"{s} {int(100 * f)}%" for s, f in OUT_SPLITS.items()) + ").", "",
          "Near-duplicate photos that are not byte-identical and carry different incident ids can still exist across "
          "splits; see `reports/data_audit/` for candidates.", ""]
    if label_conflicts:
        L += ["## Duplicates with different labels", ""] + [f"- `{' | '.join(c)}`" for c in label_conflicts[:20]] + [""]
    (args.out / "summary.md").write_text("\n".join(L))

    if args.install:
        yaml = [f"# Generated by src/make_splits.py ({VERSION})", f"path: {root.resolve()}"]
        for s in names:
            lst = root / f"{s}_{VERSION}.txt"
            lst.write_text("".join(f"./{r}\n" for r in sorted(split_imgs[s])))
            yaml.append(f"{s}: {lst.name}")
        yaml += [f"nc: {len(CLASS_NAMES)}", f"names: {CLASS_NAMES}", ""]
        (root / f"data_{VERSION}.yaml").write_text("\n".join(yaml))
        print(f"installed lists and {root / f'data_{VERSION}.yaml'}")

    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
