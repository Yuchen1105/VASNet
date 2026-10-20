"""TEST ONLY: a linear sequence stand-in exercises wiring, NOT Mamba mathematics.

Executed as a child process, so no production import is silently patched.
"""
import sys
import types
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
torch.set_num_threads(1)


class TestSequence(torch.nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.test_linear = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        # Sequence dependence makes scan/restore errors observable.
        return self.test_linear(x) + x.cumsum(1) / max(1, x.shape[1])


stub = types.ModuleType("mamba_ssm")
stub.Mamba = TestSequence
sys.modules["mamba_ssm"] = stub
from models.VASNet import VASNet, UXNET, RASEBlock, GSABlock, SCABlock

assert UXNET is VASNet
for side in (2, 3, 4, 5):
    block = RASEBlock(dim=8, slice=side).eval()
    indices = block.onion_scan_indices
    assert len(indices) == side ** 3
    assert len(set(map(tuple, indices.tolist()))) == side ** 3
    x = torch.randn(1, 8, side, side, side)
    assert torch.equal(block.restore_tensor(block.scan_tensor(x), side, side, side), x)
    with torch.inference_mode():
        a, b = block(x), block(x)
    assert torch.equal(a, b), "Canonical evaluation must not consume random scan views"
    assert a.shape == x.shape and torch.isfinite(a).all()

gsa = GSABlock(16).eval()
with torch.inference_mode():
    assert gsa(torch.randn(1, 16, 4, 4, 4)).shape == (1, 32, 4, 4, 4)
sca = SCABlock(16, 8, 8).eval()
with torch.inference_mode():
    assert sca(torch.randn(1, 16, 2, 2, 2), torch.randn(1, 8, 4, 4, 4)).shape == (1, 16, 4, 4, 4)
model = VASNet(out_chans=8)
assert set(model._modules) == {"uxnet_3d", "encoder1", "encoder2", "encoder3", "encoder4", "encoder5",
                               "decoder1", "decoder2", "decoder3", "decoder4", "decoder5", "out"}
try:
    model(torch.randn(1, 1, 64, 64, 64))
except ValueError as error:
    assert "128" in str(error)
else:
    raise AssertionError("Wrong patch shape was not rejected")
print("TEST-ONLY sequence substitute: model plumbing passed; CUDA Mamba not validated")
