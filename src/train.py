"""Train and evaluate a YOLO detector on the v1 splits.

Prerequisite: python src/make_splits.py --data "<dataset root>" --install
(writes <dataset root>/data_v1.yaml and the split lists).

Usage:
    python src/train.py --data "<dataset root>/data_v1.yaml" --model yolo11s.pt --imgsz 640 --epochs 50
    python src/train.py --data ... --fraction 0.02 --epochs 1 --name smoke   # quick check
    python src/train.py --data ... --fraction 0.5 --epochs 40 --no-test      # screening, val only

After training, the best weights are evaluated on the held-out test split and
the metrics are written to <run dir>/test_metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="path to data_v1.yaml")
    p.add_argument("--model", default="yolo11s.pt", help="pretrained weights to start from")
    p.add_argument("--imgsz", default=640, type=int)
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--patience", default=10, type=int, help="early stopping on val mAP")
    p.add_argument("--batch", default=16, type=int)
    p.add_argument("--workers", default=8, type=int)
    p.add_argument("--fraction", default=1.0, type=float, help="fraction of the train split to use")
    p.add_argument("--device", default=None, help="cuda index, mps or cpu (default: auto)")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--project", default="runs", help="output folder")
    p.add_argument("--name", default="yolo11s_640_v1")
    p.add_argument("--no-test", action="store_true", help="skip the test-split evaluation (screening runs decide on val only)")
    args = p.parse_args()

    device = args.device or pick_device()
    model = YOLO(args.model)
    model.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=args.batch,
        workers=args.workers, fraction=args.fraction, device=device, seed=args.seed, deterministic=True,
        project=args.project, name=args.name, exist_ok=True, plots=True,
    )

    run_dir = Path(model.trainer.save_dir)
    if args.no_test:
        print(f"done; test split not evaluated (--no-test). Weights: {run_dir / 'weights' / 'best.pt'}")
        return
    best = YOLO(run_dir / "weights" / "best.pt")
    m = best.val(data=args.data, split="test", imgsz=args.imgsz, batch=args.batch, device=device,
                 project=str(run_dir), name="test", exist_ok=True, plots=True)

    names = best.names
    per_class = {names[c]: {"precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                            "mAP50": float(m.box.ap50[i]), "mAP50-95": float(m.box.ap[i])}
                 for i, c in enumerate(m.box.ap_class_index)}
    out = {
        "model": args.model, "imgsz": args.imgsz, "epochs_run": model.trainer.epoch + 1, "device": device,
        "test": {"precision": float(m.box.mp), "recall": float(m.box.mr),
                 "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)},
        "per_class": per_class,
        "speed_ms_per_image": m.speed,
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
