"""Configuration and explicit, subject-disjoint dataset manifests."""
import json
from pathlib import Path


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg.get("num_classes", 0) < 2:
        raise ValueError("num_classes must include background and at least one foreground")
    if cfg.get("patch_size") != [128, 128, 128]:
        raise ValueError("The verified V32 model requires 128-cubed patches")
    return cfg


def read_manifest(path, require_labels=True, check_files=True):
    path = Path(path).resolve()
    content = json.loads(path.read_text(encoding="utf-8"))
    seen_ids, subjects, images, records = set(), {}, {}, {}
    for split in ("training", "validation", "test"):
        records[split] = []
        for raw in content.get(split, []):
            item = dict(raw)
            for key in ("id", "subject_id", "image"):
                if not isinstance(item.get(key), str) or not item[key]:
                    raise ValueError(f"Every record requires a nonempty {key}")
            if item["id"] in seen_ids:
                raise ValueError(f"Duplicate case id: {item['id']}")
            seen_ids.add(item["id"])
            owner = subjects.setdefault(item["subject_id"], split)
            if owner != split:
                raise ValueError(f"Subject leakage across splits: {item['subject_id']}")
            if require_labels and not item.get("label"):
                raise ValueError(f"Missing label: {item['id']}")
            for key in ("image", "label"):
                if item.get(key):
                    p = Path(item[key])
                    p = p if p.is_absolute() else path.parent / p
                    item[key] = str(p.resolve())
                    if key == "image":
                        previous = images.setdefault(item[key], split)
                        if previous != split:
                            raise ValueError("The same image appears in different splits")
                    if check_files and not p.is_file():
                        raise FileNotFoundError(p)
            records[split].append(item)
    if not seen_ids:
        raise ValueError("Manifest contains no cases")
    return records
