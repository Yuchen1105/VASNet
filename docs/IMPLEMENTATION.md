# Model implementation and manuscript terminology

## Verified source

The architecture comes from `models/MambaUXV32.py`, not the conference archive's earlier model. Local `VASNet_FeTA_eval.py`, `VASNet_BOBs_eval.py` and `VASNet_CBCP_eval.py` explicitly instantiate this model; the 128-cubed FTS v1 scripts do as well. Original source SHA-256: `b9a4d94fde8c41d7f98292949c943de76d1ce4fd5608abdf2d803f44163307e2`.

| Manuscript term | Historical implementation | Released public alias |
|---|---|---|
| RASE | `ux_block` | `RASEBlock` |
| GSA | `BAMBlock3D_Improved` | `GSABlock` |
| SCA | `PATSkipFusion3D` | `SCABlock` |
| Complete network | `UXNET` | `VASNet`, with `UXNET` compatibility alias |

Historical attribute names such as `uxnet_3d`, `AMFWFusion`, `encoder2` and `encoder5` remain unchanged so that state dictionaries can be loaded strictly. Class aliases do not create duplicate registered modules.

## Structure retained from the actual code

- Encoder widths: 48, 96, 192, 384; two RASE blocks at each stage. Spatial grids are 64³, 32³, 16³ and 8³ for a 128³ input. GSA expands the 384-channel bottleneck to 768 channels for the decoder.
- RASE scans concentric cuboid shells from outside to inside. Shell face planes use snake scans; intermediate planes traverse perimeters. Two channel halves are processed by separate Mamba modules (`d_state=16`, `d_conv=4`, `expand=2`). The second branch processes the outer and inner sequences separately using the same branch module.
- The inner sequence length is `(D // 2) ** 3`; its separation is an index split in the ordered sequence. This is the executable implementation, rather than a newly substituted radius formula.
- KAN branch fusion uses channel-wise softmax weighting in the source, rather than the sigmoid shorthand shown in some mathematical descriptions. This behavior is preserved.
- GSA combines channel KAN attention and separable spatial attention, then applies residual gating and grouped channel expansion.
- SCA is implemented at **three** feature skip levels (`encoder2`, `encoder3`, `encoder4`). The full-resolution `encoder1` remains a MONAI residual block. High-level features guide low-level channel/spatial calibration before refinement and concatenation. The code is not changed to add a fourth SCA just to match a schematic.
- The shallowest output block is fixed to 48 channels. Arbitrary widths or input sizes are not advertised as supported variants.

## Deliberate release changes

1. Unused profiling, logging, download helpers, stale comments and debug printing were removed. The learned operators and their parameter names were retained.
2. The unused `timm` dependency was removed. Its parameter-free DropPath behavior is implemented locally; the default drop-path rate remains zero.
3. The original source chooses a random spatial-axis view in every forward pass, including evaluation. The release defaults to a fixed canonical view during evaluation. Training remains random. `inference_scan='legacy_random'` (CLI `--scan legacy_random`) restores the original behavior. This may change predictions and must be reported when comparing historical scores.
4. An explicit 128³ input check replaces obscure failures caused by fixed scan indices. Full images are handled with sliding windows.
5. Data and training infrastructure was rewritten to remove workstation paths, private lists and conference-only utilities. That infrastructure is a portable release pipeline, not a byte-for-byte reconstruction of historical training.

No trained weights were available to establish checkpoint-level numerical equivalence. Tests check module naming and selected forward behavior using a test-only sequence substitute; see [validation](VALIDATION.md).
