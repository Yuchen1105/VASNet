import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import SimpleITK as sitk
import torch
from vasnet.config import read_config, read_manifest
from vasnet.data import normalize, validate_labels, preprocess_pair, same_grid, random_patch
from vasnet.metrics import binary_metrics
from vasnet.runtime import load_weights

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,classes", [("feta", 8), ("bobs", 4), ("cbcp", 4), ("fts", 6)])
def test_configs(name, classes):
    assert read_config(ROOT / "configs" / f"{name}.json")["num_classes"] == classes


def test_subject_leakage(tmp_path):
    record = {"id": "a", "subject_id": "s", "image": "a.nii.gz", "label": "a_label.nii.gz"}
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"training": [record], "test": [dict(record, id="b")]}))
    with pytest.raises(ValueError, match="Subject leakage"):
        read_manifest(path, check_files=False)


def test_duplicate_case(tmp_path):
    record = {"id": "a", "subject_id": "s", "image": "a", "label": "b"}
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"training": [record, record]}))
    with pytest.raises(ValueError, match="Duplicate case"):
        read_manifest(path, check_files=False)


def test_relative_path(tmp_path):
    record = {"id": "a", "subject_id": "s", "image": "image.nii.gz"}
    (tmp_path / "image.nii.gz").touch()
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"test": [record]}))
    assert read_manifest(path, require_labels=False)["test"][0]["image"] == str(tmp_path / "image.nii.gz")
    with pytest.raises(ValueError, match="Missing label"):
        read_manifest(path)


@pytest.mark.parametrize("array", [np.array([0, 1.5]), np.array([-1, 0]), np.array([0, 4]), np.array([np.nan])])
def test_invalid_labels(array):
    with pytest.raises(ValueError):
        validate_labels(array, 4)


def test_normalization_and_background():
    result = normalize(np.array([0, 2, 4, 6], dtype=float))
    assert result[0] == 0
    assert abs(result[1:].mean()) < 1e-6
    assert np.isclose(result[1:].std(), 1)
    assert not normalize(np.zeros((2, 2, 2))).any()
    with pytest.raises(ValueError):
        normalize([float("inf")])


def pair():
    a = np.zeros((7, 8, 9), np.float32)
    a[2:5, 2:6, 2:7] = 10
    image = sitk.GetImageFromArray(a)
    image.SetSpacing((1.0, 1.5, 2.0))
    image.SetOrigin((10.0, -3.0, 8.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    label = sitk.GetImageFromArray((a > 0).astype(np.uint8))
    label.CopyInformation(image)
    return image, label


def test_resampling_and_crop_physical_coordinates():
    image, label = pair()
    x, y = preprocess_pair(image, label, (1.0, 1.5, 2.0), 2, crop=True)
    assert same_grid(x, y)
    assert x.GetSize() == (5, 4, 3)
    assert np.allclose(x.GetOrigin(), image.TransformIndexToPhysicalPoint((2, 2, 2)))
    restored = sitk.Resample(y, label, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    assert np.array_equal(sitk.GetArrayFromImage(restored), sitk.GetArrayFromImage(label))


def test_resampling_preserves_orientation_and_labels():
    image, label = pair()
    x, y = preprocess_pair(image, label, (0.5, 0.5, 0.5), 2)
    assert same_grid(x, y)
    assert x.GetSize() == (17, 22, 25)
    assert x.GetDirection() == image.GetDirection()
    assert x.GetOrigin() == image.GetOrigin()
    assert set(np.unique(sitk.GetArrayFromImage(y))) <= {0, 1}


def test_grid_mismatch_rejected():
    image, label = pair()
    label.SetOrigin((0, 0, 0))
    with pytest.raises(ValueError, match="share"):
        preprocess_pair(image, label, (1, 1, 1), 2)


def test_patch_padding_and_alignment():
    y = np.ones((3, 4, 5), dtype=np.int64)
    x, target = random_patch(y.astype(np.float32), y, (8, 8, 8))
    assert x.shape == (1, 8, 8, 8)
    assert target.sum() == 60
    assert np.array_equal(x[0], target)
    assert x.flags.c_contiguous and target.flags.c_contiguous


def test_physical_asd():
    a, b = np.zeros((5, 5, 5), bool), np.zeros((5, 5, 5), bool)
    a[1, 2, 2], b[2, 2, 2] = True, True
    metrics = binary_metrics(a, b, (3, 2, 1))
    assert metrics["asd_mm"] == 3
    assert metrics["dice"] == metrics["iou"] == 0
    assert binary_metrics(a, a, (3, 2, 1))["dice"] == 1


def test_empty_mask_conventions():
    empty = np.zeros((3, 3, 3), bool)
    full = np.ones_like(empty)
    assert binary_metrics(empty, empty, (1, 1, 1))["status"] == "both_empty"
    one_empty = binary_metrics(full, empty, (1, 1, 1))
    assert one_empty["asd_mm"] is None and one_empty["dice"] == 0


def test_strict_checkpoint_and_ddp(tmp_path):
    model = torch.nn.Linear(3, 2)
    path = tmp_path / "weights.pt"
    torch.save({"state_dict": {"module." + k: v for k, v in model.state_dict().items()}}, path)
    other = torch.nn.Linear(3, 2)
    load_weights(other, path)
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items())
    with pytest.raises(RuntimeError):
        load_weights(torch.nn.Linear(3, 4), path)
    torch.save({"weight": model.weight}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_weights(model, path)


@pytest.mark.parametrize("script", ["train", "predict", "preprocess", "evaluate", "validate_manifest", "smoke_cuda"])
def test_cli_help(script):
    result = subprocess.run([sys.executable, str(ROOT / "Code" / f"{script}.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_model_plumbing_in_isolated_subprocess():
    result = subprocess.run([sys.executable, str(ROOT / "tests/model_plumbing_check.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "plumbing passed" in result.stdout
