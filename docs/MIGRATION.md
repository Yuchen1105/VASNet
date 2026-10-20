# Conference archive to VASNet journal release

The input `VSMamba-main (1).zip` was retained unchanged. Its scripts were inspected for organization, but the journal architecture was taken from the verified local V32 code used by the VASNet evaluations.

| Earlier artifact / behavior | Journal release |
|---|---|
| VSMamba naming and conference README | VASNet repository; manuscript title, authors and submission status |
| Earlier model / experiment variants | Actual `MambaUXV32` architecture, public `VASNet` class |
| Workstation drive letters, GPU indices and embedded split lists | Manifest paths and CLI device selection |
| Numerous unrelated model implementations and scripts | One model, four dataset configs and focused runnable entry points |
| Conference-specific inference utilities | Gaussian sliding windows and argmax |
| Scattered evaluation calculations | Auditable case/class metrics with physical spacing |
| Assumed training protocol | Explicit manuscript/local optimizer profiles and documented discrepancies |
| Unqualified result claims | Manuscript table clearly distinguished from newly measured results |

Historical imports can migrate from `from models.MambaUXV32 import UXNET` to `from models.VASNet import VASNet` when `Code` is on `PYTHONPATH`. The `UXNET` name is also exported by `models.VASNet` for compatibility. A historical state dictionary should be loaded with `vasnet.runtime.load_weights`; it is checked strictly. This expectation is based on preserved registered names and structural checks; validation with a genuine historical V32 checkpoint remains outstanding.

The new repository excludes patient data, institutional split IDs, local checkpoints, logs, credentials and unrelated experiments. Upstream source files on the preparation computer were not edited. No remote repository was created or changed by this preparation.
