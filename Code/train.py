"""Portable single-device training. See docs/REPRODUCIBILITY.md for protocol limits."""
import argparse
import json
from pathlib import Path
from vasnet.config import read_config, read_manifest


def main():
    parser = argparse.ArgumentParser(description="Train the local-source VASNet with an explicit optimization profile")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New directory, or an existing run only with --resume")
    parser.add_argument("--profile", required=True, choices=["paper_adam", "local_sgd"])
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--steps-per-epoch", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--resume", help="Resume a release-format last.pt, keeping optimizer and scheduler state")
    args = parser.parse_args()
    if min(args.epochs, args.steps_per_epoch, args.validate_every) < 1 or args.batch_size < 2:
        raise ValueError("Epochs, steps and validation interval must be positive; SRM training requires batch >= 2")
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from vasnet.data import load_pair, random_patch
    from vasnet.runtime import seed_all, build_model, load_weights, window_logits
    cfg, splits = read_config(args.config), read_manifest(args.manifest)
    if not splits["training"] or not splits["validation"]:
        raise ValueError("Training and validation splits must both be nonempty")
    output = Path(args.output).resolve()
    if args.resume and not output.is_dir():
        raise ValueError("Resume output must be the existing run directory")
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    model = build_model(cfg).to(args.device)
    if args.profile == "paper_adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99, nesterov=True, weight_decay=3e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - min(epoch, args.epochs) / args.epochs) ** 0.9)
    start, best = 0, -1.0
    if args.resume:
        checkpoint = load_weights(model, args.resume)
        if checkpoint.get("config") != cfg:
            raise ValueError("Resume configuration differs from checkpoint")
        if checkpoint.get("manifest") != splits:
            raise ValueError("Resume manifest differs from checkpoint")
        for key in ("profile", "epochs", "steps_per_epoch", "batch_size", "seed", "validate_every"):
            if checkpoint["training"][key] != vars(args)[key]:
                raise ValueError(f"Resume cannot change {key}")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best_validation_dice"]
        # Stored RNG states use only safe tensor/basic types for weights_only loading.
        random.setstate(checkpoint["rng_python"])
        ns = checkpoint["rng_numpy"]
        np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        torch.set_rng_state(checkpoint["rng_torch"])
        if torch.cuda.is_available() and checkpoint["rng_cuda"]:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
    settings = {"config": cfg, "training": vars(args), "manifest": splits,
                "note": "Portable release pipeline; not a recovered historical experiment log."}
    (output / ("resume_settings.json" if args.resume else "settings.json")).write_text(json.dumps(settings, indent=2), encoding="utf-8")
    # Caching all volumes is deliberately avoided. Adjust IO/caching locally for large cohorts.
    for epoch in range(start, args.epochs):
        model.train()
        total_loss = 0.0
        for _ in range(args.steps_per_epoch):
            batch = [random_patch(*load_pair(random.choice(splits["training"]), cfg["num_classes"]), cfg["patch_size"])
                     for _ in range(args.batch_size)]
            x = torch.from_numpy(np.stack([b[0] for b in batch])).to(args.device)
            y = torch.from_numpy(np.stack([b[1] for b in batch])).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        validation_dice = None
        improved = False
        if (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.epochs:
            model.eval()
            scores = []
            with torch.inference_mode():
                for case in splits["validation"]:
                    image, target = load_pair(case, cfg["num_classes"])
                    prediction = window_logits(model, torch.from_numpy(image[None, None]), args.device).argmax(1)[0].numpy()
                    for cls in range(1, cfg["num_classes"]):
                        p, t = prediction == cls, target == cls
                        denom = p.sum() + t.sum()
                        scores.append(float(2 * (p & t).sum() / denom) if denom else 1.0)
            validation_dice = float(np.mean(scores))
            improved = validation_dice > best
            best = max(best, validation_dice)
        np_state = np.random.get_state()
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch, "config": cfg,
                      "training": vars(args), "manifest": splits, "best_validation_dice": best,
                      "rng_python": random.getstate(), "rng_numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
                      "rng_torch": torch.get_rng_state(),
                      "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        temporary = output / "last.tmp.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "last.pt")
        if improved:
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch,
                        "validation_dice": validation_dice, "inference_scan": "canonical"}, output / "best.pt")
        record = {"epoch": epoch, "loss": total_loss / args.steps_per_epoch, "validation_dice": validation_dice,
                  "next_learning_rate": optimizer.param_groups[0]["lr"]}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
