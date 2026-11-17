"""Prepare the Roboflow "Pothole Detection Dataset" (Aegis, v2, 1,482 images) as an external test set.

The paper (AlElaiwi et al., 2022) re-tested its model on this dataset. This script takes the
YOLO-format export (train/valid/test folders, class 0 = pothole), pools all splits, and copies
them into <out>/images and <out>/labels with the class id remapped to our pothole id (2).
It also writes <out>/list.txt for `src/eval_ci.py predict`.

Usage:
    python src/prep_external_pothole.py <roboflow_export_dir> <out_dir>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

OUR_POTHOLE_ID = 2
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()

    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "labels").mkdir(parents=True, exist_ok=True)
    listed, n_box, bad = [], 0, 0
    for img in sorted(p for p in a.src.rglob("*") if p.suffix.lower() in IMG_EXT and "images" in p.parts):
        lbl = Path(*["labels" if q == "images" else q for q in img.parts]).with_suffix(".txt")
        rows = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                f = line.split()
                if len(f) != 5:  # polygons or malformed rows are skipped and counted
                    bad += bool(f)
                    continue
                if f[0] != "0":
                    raise SystemExit(f"unexpected class {f[0]} in {lbl}")
                rows.append(" ".join([str(OUR_POTHOLE_ID), *f[1:]]))
        name = f"{img.parent.parent.name}_{img.name}"  # keep names unique across splits
        dst = a.out / "images" / name
        shutil.copy2(img, dst)
        (a.out / "labels" / Path(name).with_suffix(".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
        listed.append(str(dst.resolve()))
        n_box += len(rows)
    (a.out / "list.txt").write_text("\n".join(listed) + "\n")
    print(f"{len(listed)} images, {n_box} pothole boxes, {bad} non-box rows skipped -> {a.out}/list.txt")


if __name__ == "__main__":
    main()
