import argparse
import json
from pathlib import Path
from vasnet.config import read_config, read_manifest


def main():
    parser = argparse.ArgumentParser(description="Resample MRI/labels to an explicitly chosen physical grid")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New directory; originals are never overwritten")
    parser.add_argument("--spacing", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--crop-nonzero", action="store_true")
    args = parser.parse_args()
    import SimpleITK as sitk
    from vasnet.data import read_image, preprocess_pair
    config, splits = read_config(args.config), read_manifest(args.manifest)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"preprocessing": {"spacing_xyz": args.spacing, "crop_nonzero": args.crop_nonzero,
                                "source_manifest": str(Path(args.manifest).resolve())}}
    for split, cases in splits.items():
        result[split] = []
        for case in cases:
            image, label = preprocess_pair(read_image(case["image"]), read_image(case["label"]),
                                           args.spacing, config["num_classes"], args.crop_nonzero)
            # IDs are metadata, never used as filesystem paths.
            stem = f"case_{sum(len(result.get(k, [])) for k in splits):05d}"
            im_path, lab_path = f"{stem}_image.nii.gz", f"{stem}_label.nii.gz"
            sitk.WriteImage(image, str(output / im_path))
            sitk.WriteImage(label, str(output / lab_path))
            result[split].append({"id": case["id"], "subject_id": case["subject_id"],
                                  "image": im_path, "label": lab_path})
    (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
