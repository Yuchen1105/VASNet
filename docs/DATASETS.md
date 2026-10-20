# Datasets and labels

| Configuration | Dataset | Output channels, including background |
|---|---|---:|
| `configs/feta.json` | FeTA | 8 |
| `configs/bobs.json` | BOBs | 4 |
| `configs/cbcp.json` | CBCP | 4 |
| `configs/fts.json` | Institutional fetal tissue segmentation (FTS) | 6 |

Counts are taken from the local V32 experiment entry points. Label IDs must be contiguous integers starting at zero; images are single-channel scalar 3-D volumes. No automatic relabeling is performed. Establish the tissue-to-ID mapping from the authorized dataset/preprocessing release, especially for FTS. Configuration files intentionally use numeric labels rather than inventing tissue correspondences that were not independently verified.

Download or access data through its original provider and applicable institutional permissions. The manuscript's FTS cohort contains 96 fetal MRI cases, but those images, annotations and patient identifiers are not distributed here. This repository does not imply access permission or a clinical release.

Create your own manifest using `data/manifest.example.json`. Cases and subjects must be disjoint across training, validation and test; multiple scans of the same subject belong in the same split. A record requires a unique case ID, subject ID, image path and (for training/evaluation) label path. Prediction accepts image-only records. Relative paths are anchored to the manifest location.

The validator checks labels and matching image/label grids. Preserve original images to restore cropped/resampled predictions. SimpleITK spacing is x,y,z; NumPy mask arrays are z,y,x. The metric implementation reverses spacing accordingly. No automatic orientation canonicalization or registration is applied; both members of a pair must already be aligned.

The preprocessing CLI writes resampled intensity images without z-score normalization. Runtime loading applies foreground z-score normalization once. Optional nonzero cropping retains physical coordinates using SimpleITK ROI extraction. For all-zero images the normalized output is zero. Predictions outside a cropped physical field are filled with background when restored to the original grid.
