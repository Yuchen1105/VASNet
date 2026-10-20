"""Single-channel 3-D MRI IO, physical-space preprocessing and patch sampling."""
import numpy as np
import SimpleITK as sitk


def read_image(path):
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"Expected a scalar 3-D volume: {path}")
    return image


def same_grid(a, b):
    return (a.GetSize() == b.GetSize() and
            all(np.allclose(x, y, atol=1e-5, rtol=0) for x, y in (
                (a.GetSpacing(), b.GetSpacing()), (a.GetOrigin(), b.GetOrigin()),
                (a.GetDirection(), b.GetDirection()))))


def validate_labels(array, num_classes):
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError("Labels must contain finite integers")
    if array.min() < 0 or array.max() >= num_classes:
        raise ValueError(f"Labels must be in [0, {num_classes - 1}]")


def normalize(array):
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or infinite values")
    mask = array != 0
    output = np.zeros_like(array)
    if mask.any():
        values = array[mask]
        output[mask] = (values - values.mean()) / max(float(values.std()), 1e-8)
    return output


def preprocess_pair(image, label, spacing_xyz, num_classes, crop=False):
    if not same_grid(image, label):
        raise ValueError("Image and label must share size, spacing, origin and direction")
    validate_labels(sitk.GetArrayFromImage(label), num_classes)
    spacing = np.asarray(spacing_xyz, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Spacing must contain three positive finite values in x,y,z order")
    # Preserve endpoint extent, orientation and origin. One reference defines both grids.
    size = np.maximum(1, np.rint((np.array(image.GetSize()) - 1) * image.GetSpacing() / spacing).astype(int) + 1)
    reference = sitk.Image([int(x) for x in size], sitk.sitkFloat32)
    reference.SetSpacing(spacing.tolist())
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    image = sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    label = sitk.Resample(label, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    if crop:
        coords = np.argwhere(sitk.GetArrayFromImage(image) != 0)
        if len(coords):
            lo, hi = coords.min(0), coords.max(0) + 1
            image = sitk.RegionOfInterest(image, (hi - lo)[::-1].tolist(), lo[::-1].tolist())
            label = sitk.RegionOfInterest(label, (hi - lo)[::-1].tolist(), lo[::-1].tolist())
    return image, label


def load_pair(record, num_classes):
    image, label = read_image(record["image"]), read_image(record["label"])
    if not same_grid(image, label):
        raise ValueError(f"Mismatched image/label grid: {record['id']}")
    y = sitk.GetArrayFromImage(label)
    validate_labels(y, num_classes)
    return normalize(sitk.GetArrayFromImage(image)), y.astype(np.int64)


def random_patch(image, label, patch_size, foreground_probability=0.5):
    padding = [(0, max(0, p - s)) for p, s in zip(patch_size, label.shape)]
    image, label = np.pad(image, padding), np.pad(label, padding)
    foreground = np.argwhere(label > 0)
    if len(foreground) and np.random.random() < foreground_probability:
        center = foreground[np.random.randint(len(foreground))]
        start = [int(np.clip(c - p // 2, 0, s - p)) for c, p, s in zip(center, patch_size, label.shape)]
    else:
        start = [np.random.randint(s - p + 1) for s, p in zip(label.shape, patch_size)]
    region = tuple(slice(a, a + p) for a, p in zip(start, patch_size))
    x, y = image[region], label[region]
    for axis in range(3):
        if np.random.random() < 0.5:
            x, y = np.flip(x, axis), np.flip(y, axis)
    return np.ascontiguousarray(x[None]), np.ascontiguousarray(y)
