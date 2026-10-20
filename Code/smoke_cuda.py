"""Run only in a real Linux/CUDA/Mamba environment, never with a substitute layer."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Real Mamba forward/backward smoke test; requires substantial GPU memory")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backward", action="store_true", help="Uses batch 2 for pooled SRM BatchNorm")
    args = parser.parse_args()
    import torch
    from models import VASNet
    model = VASNet(out_chans=args.classes).to(args.device)
    model.train(args.backward)
    x = torch.randn(2 if args.backward else 1, 1, 128, 128, 128, device=args.device)
    if args.backward:
        output = model(x)
        output.square().mean().backward()
    else:
        with torch.inference_mode():
            output = model(x)
    assert output.shape == (x.shape[0], args.classes, 128, 128, 128)
    assert torch.isfinite(output).all()
    print({"shape": list(output.shape), "parameters": sum(p.numel() for p in model.parameters()),
           "backward": args.backward})


if __name__ == "__main__":
    main()
