# Reproducibility and unresolved protocol differences

This repository preserves the actual local journal model and makes its execution interfaces portable. It does **not** imply that all settings in the submitted manuscript have been reconciled with every local experiment.

| Item | Submitted manuscript / local evidence | Release policy |
|---|---|---|
| Optimizer | Manuscript: Adam, 1e-4, cosine. Local V32 scripts: SGD, 1e-2, momentum .99, Nesterov, weight decay 3e-5, polynomial decay | Explicit `paper_adam` and `local_sgd` profiles |
| Duration | Local FeTA/BOBs/CBCP scripts use 1000 epochs; FTS v1 defaults to 800. A single journal duration was not established | Require `--epochs` and `--steps-per-epoch` |
| Input | Verified journal evaluation uses 128³ patches; another local FTS variant uses 64³ | Release supports the 128³ source only |
| Batch | Batch 2 in manuscript; pooled SRM BatchNorm cannot train with batch 1 | Enforce batch >= 2 |
| Split | Manuscript describes 8:1:1; recovered BOBs/CBCP test lists contain 7/20 entries and cannot establish one consistent published split | Require external, subject-disjoint manifests; no private IDs shipped |
| Orientation during evaluation | Manuscript describes a fixed orientation; source selects random views even in eval | Fixed by default, original behavior available as `legacy_random` |
| Preprocessing | Historical spacing and dataset preparation differ across scripts | Explicit spacing; no universal automatic spacing claim |
| Postprocessing | Some conference utilities contain RankSEG; journal evaluation uses multiclass outputs | Argmax only; no RankSEG dependency |
| Test-time augmentation | Local evaluation commonly uses mirrored views | Optional `--mirror`, recorded in prediction metadata |
| Original weights | Verified V32 checkpoints were not found in the inspected local weight locations | No weights bundled; strict loader only |

## Newly organized training behavior

The loader normalizes nonzero voxels, pads volumes smaller than the patch, samples foreground-centered patches with probability 0.5 and applies independent random axis flips. Other patches are sampled uniformly. Cases are sampled uniformly with replacement. Cross-entropy supervises the single final output. Validation uses the complete held-out volumes, Gaussian sliding windows and no mirroring; model selection uses foreground Dice averaged over case/class pairs. This is stated explicitly so it can be reviewed against the intended final protocol.

The run directory records configuration, resolved manifest, epoch loss, validation Dice and checkpoints. Resume retains Python/NumPy/PyTorch RNG states and optimizer/scheduler state. It assumes the same dataset files, software, device arrangement and settings; it is not a guarantee of bitwise reproducibility across CUDA platforms.

## Metrics

Dice and IoU are fractions in JSON, with background excluded. Overall means weight every case/foreground-class pair equally. If both masks are empty, Dice/IoU are 1 and ASD is 0. If exactly one is empty, Dice/IoU are 0 and ASD is undefined (`null`). Undefined ASD counts are reported and are never silently replaced by a finite distance.

ASD defaults to the pooled bidirectional surface distance, using six-connected surface extraction and physical voxel spacing. `--asd-mode directed` computes prediction-to-reference ASD for comparisons with a directed implementation. Historical scripts and literature may use different surface distance or absent-class conventions; the default metric must not be assumed numerically identical to every paper table. The JSON includes per-case/per-class values to audit aggregation.

## Before claiming reproduction

Confirm the final optimizer and sampling protocol, approved subject splits, class mappings, spacing, original-grid restoration, scan orientation, test-time augmentation, metric conventions and the exact checkpoint used. Run the real CUDA forward/backward check and the full experiment in that environment. Compare held-out predictions and metrics, rather than using CPU plumbing tests as evidence of model accuracy.
