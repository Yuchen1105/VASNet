"""Synthetic IO integration and training infrastructure with an explicit toy model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]


def invoke(script, *args):
    result = subprocess.run([sys.executable, str(ROOT / "Code" / script), *map(str, args)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def synthetic_manifest(tmp_path):
    array = np.zeros((8, 8, 8), dtype=np.float32)
    array[2:6, 2:6, 2:6] = np.arange(64).reshape(4, 4, 4) + 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1, 1.5, 2))
    image.SetOrigin((4, 5, 6))
    mask = sitk.GetImageFromArray((array > 0).astype(np.uint8))
    mask.CopyInformation(image)
    splits = {}
    for i, split in enumerate(("training", "validation", "test")):
        sitk.WriteImage(image, str(tmp_path / f"image{i}.nii.gz"))
        sitk.WriteImage(mask, str(tmp_path / f"label{i}.nii.gz"))
        splits[split] = [{"id": f"case{i}", "subject_id": f"subject{i}",
                          "image": f"image{i}.nii.gz", "label": f"label{i}.nii.gz"}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(splits))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"num_classes": 2, "patch_size": [128, 128, 128]}))
    return manifest, config


def test_preprocess_and_evaluate_cli(tmp_path):
    manifest, config = synthetic_manifest(tmp_path)
    invoke("validate_manifest.py", "--manifest", manifest, "--config", config)
    output = tmp_path / "prepared"
    invoke("preprocess.py", "--manifest", manifest, "--config", config,
           "--output", output, "--spacing", 1, 1.5, 2, "--crop-nonzero")
    processed = json.loads((output / "manifest.json").read_text())
    assert len({r["image"] for split in ("training", "validation", "test") for r in processed[split]}) == 3
    test = processed["test"][0]
    predictions = output / "predictions.json"
    predictions.write_text(json.dumps({"predictions": [{"id": test["id"], "prediction": test["label"]}]}))
    result = tmp_path / "metrics.json"
    invoke("evaluate.py", "--manifest", output / "manifest.json", "--config", config,
           "--predictions", predictions, "--output", result)
    score = json.loads(result.read_text())["overall"]
    assert score["dice"] == score["iou"] == 1.0
    assert score["asd_mm_defined_only"] == 0.0


def test_toy_training_and_resume(tmp_path):
    manifest, config = synthetic_manifest(tmp_path)
    # Subprocess injection is visible and confined to this infrastructure test.
    # A two-channel 1x1 Conv3d replaces the WHOLE model, not a production SSM.
    script = '''
import sys, torch
torch.set_num_threads(1)
from vasnet import runtime
runtime.build_model = lambda cfg, *args: torch.nn.Conv3d(1, cfg["num_classes"], 1)
import train
train.main()
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / "Code"))
    run = tmp_path / "run"
    args = ["--config", str(config), "--manifest", str(manifest), "--output", str(run),
            "--profile", "paper_adam", "--epochs", "1", "--steps-per-epoch", "1", "--device", "cpu"]
    for tail in ([], ["--resume", str(run / "last.pt")]):
        result = subprocess.run([sys.executable, "-c", script, *args, *tail], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    assert (run / "last.pt").is_file() and (run / "best.pt").is_file()
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert len(history) == 1 and np.isfinite(history[0]["loss"])
