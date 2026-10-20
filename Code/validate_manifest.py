import argparse
from vasnet.config import read_config, read_manifest


def main():
    parser = argparse.ArgumentParser(description="Validate explicit patient-disjoint data splits and volume labels")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--headers-only", action="store_true", help="Check paths and splits without reading voxel arrays")
    args = parser.parse_args()
    cfg, splits = read_config(args.config), read_manifest(args.manifest)
    if not args.headers_only:
        from vasnet.data import load_pair
        for cases in splits.values():
            for case in cases:
                load_pair(case, cfg["num_classes"])
    print({split: len(cases) for split, cases in splits.items()})


if __name__ == "__main__":
    main()
