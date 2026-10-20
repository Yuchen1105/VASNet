import argparse
import json
from pathlib import Path
from vasnet.config import read_config, read_manifest


def main():
    parser = argparse.ArgumentParser(description="Evaluate NIfTI masks using physical-space surface distances")
    parser.add_argument("--manifest", required=True, help="Ground truth manifest, matching the prediction grid")
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True, help="predictions.json produced by predict.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["training", "validation", "test"], default="test")
    parser.add_argument("--asd-mode", choices=["symmetric", "directed"], default="symmetric")
    args = parser.parse_args()
    import numpy as np
    import SimpleITK as sitk
    from vasnet.data import read_image, same_grid, validate_labels
    from vasnet.metrics import binary_metrics
    cfg, splits = read_config(args.config), read_manifest(args.manifest)
    pred_path = Path(args.predictions).resolve()
    entries = json.loads(pred_path.read_text(encoding="utf-8"))["predictions"]
    predictions = {p["id"]: pred_path.parent / p["prediction"] for p in entries}
    if len(predictions) != len(entries) or set(predictions) != {r["id"] for r in splits[args.split]}:
        raise ValueError("Prediction IDs must exactly match the selected split without duplicates")
    rows = []
    for case in splits[args.split]:
        p, t = read_image(predictions[case["id"]]), read_image(case["label"])
        if not same_grid(p, t):
            raise ValueError(f"Grid mismatch for {case['id']}; restore to the reference grid first")
        pa, ta = sitk.GetArrayFromImage(p), sitk.GetArrayFromImage(t)
        validate_labels(pa, cfg["num_classes"])
        validate_labels(ta, cfg["num_classes"])
        for cls in range(1, cfg["num_classes"]):
            rows.append(dict(id=case["id"], class_id=cls,
                             **binary_metrics(pa == cls, ta == cls, t.GetSpacing()[::-1], args.asd_mode)))
    if not rows:
        raise ValueError("No cases to evaluate")
    def aggregate(group):
        valid_asd = [r["asd_mm"] for r in group if r["asd_mm"] is not None]
        return {"dice": float(np.mean([r["dice"] for r in group])),
                "iou": float(np.mean([r["iou"] for r in group])),
                "asd_mm_defined_only": float(np.mean(valid_asd)) if valid_asd else None,
                "asd_undefined": len(group) - len(valid_asd), "count": len(group)}
    result = {"asd_mode": args.asd_mode, "overall": aggregate(rows),
              "per_class": {str(c): aggregate([r for r in rows if r["class_id"] == c])
                            for c in range(1, cfg["num_classes"])}, "cases": rows}
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
