"""Foreground overlap and spacing-aware surface distance; no silent empty-mask drops."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def binary_metrics(prediction, target, spacing_zyx, asd_mode="symmetric"):
    p, t = np.asarray(prediction, bool), np.asarray(target, bool)
    if p.shape != t.shape:
        raise ValueError("Prediction and target shapes differ")
    if asd_mode not in ("symmetric", "directed"):
        raise ValueError("Unknown ASD mode")
    if len(spacing_zyx) != p.ndim or any(not np.isfinite(s) or s <= 0 for s in spacing_zyx):
        raise ValueError("Spacing must match mask dimensions and be positive")
    npred, ntrue = int(p.sum()), int(t.sum())
    if not npred and not ntrue:
        return dict(dice=1.0, iou=1.0, asd_mm=0.0, status="both_empty")
    intersection = int((p & t).sum())
    dice = 2 * intersection / (npred + ntrue)
    iou = intersection / (npred + ntrue - intersection)
    if not npred or not ntrue:
        return dict(dice=dice, iou=iou, asd_mm=None, status="one_empty")
    structure = generate_binary_structure(p.ndim, 1)
    ps = p ^ binary_erosion(p, structure=structure, border_value=0)
    ts = t ^ binary_erosion(t, structure=structure, border_value=0)
    p_to_t = distance_transform_edt(~ts, sampling=spacing_zyx)[ps]
    if asd_mode == "symmetric":
        t_to_p = distance_transform_edt(~ps, sampling=spacing_zyx)[ts]
        asd = float(np.concatenate((p_to_t, t_to_p)).mean())
    else:
        asd = float(p_to_t.mean())
    return dict(dice=float(dice), iou=float(iou), asd_mm=asd, status="valid")
