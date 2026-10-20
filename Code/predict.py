import argparse
import json
from pathlib import Path
from vasnet.config import read_config, read_manifest


def main():
    parser = argparse.ArgumentParser(description="VASNet Gaussian sliding-window prediction")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["training", "validation", "test"], default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scan", choices=["canonical", "legacy_random"], default="canonical")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-manifest", help="Original grid manifest for nearest-neighbor restoration")
    args = parser.parse_args()
    import numpy as np
    import torch
    import SimpleITK as sitk
    from vasnet.data import read_image, normalize
    from vasnet.runtime import seed_all, build_model, load_weights, window_logits
    cfg, splits = read_config(args.config), read_manifest(args.manifest, require_labels=False)
    if not splits[args.split]:
        raise ValueError("Selected split is empty")
    references = None
    if args.reference_manifest:
        original = read_manifest(args.reference_manifest, require_labels=False)
        references = {r["id"]: r for r in original[args.split]}
        if set(references) != {r["id"] for r in splits[args.split]}:
            raise ValueError("Reference and input case IDs must match within the selected split")
    seed_all(args.seed)
    model = build_model(cfg, args.scan).to(args.device).eval()
    load_weights(model, args.checkpoint)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    index = {"settings": vars(args), "predictions": []}
    with torch.inference_mode():
        for number, case in enumerate(splits[args.split]):
            image = read_image(case["image"])
            x = torch.from_numpy(normalize(sitk.GetArrayFromImage(image))[None, None])
            logits = window_logits(model, x, args.device, args.mirror, sw_batch_size=args.sw_batch_size)
            pred = sitk.GetImageFromArray(logits.argmax(1)[0].numpy().astype(np.uint16))
            pred.CopyInformation(image)
            if references is not None:
                ref = read_image(references[case["id"]]["image"])
                pred = sitk.Resample(pred, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
            filename = f"prediction_{number:05d}.nii.gz"
            sitk.WriteImage(pred, str(output / filename))
            index["predictions"].append({"id": case["id"], "prediction": filename})
    (output / "predictions.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
