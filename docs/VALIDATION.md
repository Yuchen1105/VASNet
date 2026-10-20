# Validation performed for this release

Preparation date: 2026-10-19. Platform: Windows, Python 3.10, isolated CPU test environment with PyTorch 2.4.0+cpu, MONAI 1.3.2, NumPy 1.26.4, SciPy 1.13.1 and SimpleITK 2.4.0.

## Automated checks

`python -m pytest -q`: **28 passed**.

The suite checks four dataset configurations, subject-disjoint manifests, duplicate cases, relative path resolution, required labels, invalid label IDs, foreground normalization, physical-coordinate preservation during resampling/cropping, original-grid restoration, nearest-neighbor labels, patch padding/alignment, spacing-aware ASD, empty-mask conventions, strict checkpoint loading and DDP prefix removal. All six CLI help commands are exercised.

Synthetic integration checks write real NIfTI volumes, run the preprocessing and evaluation CLIs, and verify perfect-mask metrics. A separate test runs one training epoch and checkpoint resume with an explicitly injected **toy Conv3d model**. This exercises the newly written training/checkpoint infrastructure, not VASNet optimization or accuracy.

An isolated child process installs a **test-only linear sequence stand-in** to check shell-index uniqueness, scan/restore round trips for side lengths 2–5, fixed-view evaluation repeatability, RASE/GSA/SCA output shapes, model module names and rejection of 64³ patches. No production code imports that stand-in.

## Comparison with the actual local source

The reference was generated directly from the original `MambaUXV32.py` classes, removing only unused utilities, inactive strings and printing. The same test-only sequence stand-in was used in both reference and release:

- All **615** state-dictionary entries in that test-substitute model had identical names and shapes.
- Strict loading from reference to release preserved all tensor values exactly.
- A RASE block in `legacy_random` mode produced bitwise-equal outputs to the reference for five Python RNG seeds after loading the same state.
- Independent initialization was not bitwise-identical for 20 entries. Consequently the comparison uses an explicitly shared loaded state rather than assuming that resetting an RNG makes KAN least-squares initialization bitwise reproducible.

These checks support the naming/graph migration under the test substitute. They **do not establish the state schema of the real Mamba implementation**, checkpoint-level prediction equivalence or equivalence to the submitted metrics. The original source checksum and reviewed experiment entry points are retained in `SOURCE_PROVENANCE.json`.

## Outstanding execution checks

No verified trained V32 checkpoint was found in the inspected local weight locations. Real Mamba CUDA forward/backward, GPU memory requirements, full training, actual dataset evaluation, and the submitted numerical results were **not rerun**. `Code/smoke_cuda.py` provides the real-model GPU smoke test for the target environment. The pinned Linux/CUDA installation remains to be validated on that environment.

The supplied archive and local source files were read without modification. The release contains no patient volumes, original private split lists, trained weights, workstation-specific absolute paths or credentials. Figures are copied from the supplied manuscript material. No remote GitHub action was performed.
