"""
CLI for training TinyDerainNet (src/lightweight_derain_net.py) on
procedurally-generated synthetic rain data -- no external dataset required.

IMPORTANT CONTEXT (read before spending compute on this): in this project's
own development/testing (1 CPU core, limited wall-clock budget), multiple
training configurations (weighted loss, stability regularization, learning
rate sweeps, fixed-scene-pool data) were tried and NONE produced a model
that clearly beat the trivial "do nothing" baseline on held-out synthetic
data -- let alone the classical multi-frame TemporalDerain method, which
reliably gets +4-4.5 dB. This is an honest, measured negative result, not a
guess. Likely causes: single-image deraining is a fundamentally harder,
less-constrained problem than the multi-frame approach (no temporal
redundancy to exploit), and a 35K-parameter network trained for a few
thousand total samples on 1 CPU core is a very small compute budget for this
task.

This script is provided so you can pick up where that left off with more
compute (more epochs/samples, a multi-core machine, a GPU, or a larger
model) -- not because the current recipe is known to work well. It always
prints a baseline comparison so you can see immediately whether a given
training run actually achieved anything.

Usage:
    python train_cnn.py --epochs 30 --samples-per-epoch 400 --patch-size 128 \
        --checkpoint ../models_train/tiny_derain.pt
    python train_cnn.py --resume-from ../models_train/tiny_derain.pt --epochs 20  # continue training
    python train_cnn.py --export-onnx ../models/tiny_derain.onnx --checkpoint ../models_train/tiny_derain.pt --export-only
"""
import argparse
import numpy as np

from lightweight_derain_net import train, export_onnx, SyntheticDerainDataset, TORCH_AVAILABLE


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse <= 1e-10 else 10.0 * np.log10((255.0 ** 2) / mse)


def baseline_psnr(patch_size, val_samples, seed=999):
    """PSNR of doing nothing (rainy vs clean) on the same validation
    distribution used during training -- the number any trained checkpoint
    needs to beat to be worth using at all."""
    val_ds = SyntheticDerainDataset(patch_size=patch_size, length=val_samples, seed=seed)
    psnrs = []
    for i in range(len(val_ds)):
        rainy_t, clean_t, _ = val_ds[i]
        rainy_img = rainy_t.numpy().transpose(1, 2, 0) * 255.0
        clean_img = clean_t.numpy().transpose(1, 2, 0) * 255.0
        psnrs.append(_psnr(rainy_img, clean_img))
    return float(np.mean(psnrs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--samples-per-epoch", type=int, default=400)
    ap.add_argument("--val-samples", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--checkpoint", default="../models_train/tiny_derain.pt")
    ap.add_argument("--resume-from", default=None, help="Path to an existing checkpoint to continue training from")
    ap.add_argument("--export-onnx", default=None, help="If set, export to this .onnx path after training")
    ap.add_argument("--export-only", action="store_true",
                     help="Skip training entirely, just export --checkpoint to --export-onnx")
    args = ap.parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit("torch is required for training. pip install -r requirements-train.txt")

    if args.export_only:
        if not args.export_onnx:
            raise SystemExit("--export-only requires --export-onnx <path>")
        export_onnx(args.checkpoint, out_path=args.export_onnx, img_size=args.patch_size)
        return

    base_psnr = baseline_psnr(args.patch_size, args.val_samples)
    print(f"Baseline (no processing) PSNR on this validation set: {base_psnr:.2f} dB")
    print(f"-> training only helps if the model clearly and consistently beats this.\n")

    model, history = train(
        epochs=args.epochs, batch_size=args.batch_size, patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch, val_samples=args.val_samples,
        lr=args.lr, checkpoint_path=args.checkpoint, resume_from=args.resume_from,
    )

    best_psnr = max(h["val_psnr"] for h in history)
    gain = best_psnr - base_psnr
    print(f"\nBest validation PSNR this run: {best_psnr:.2f} dB "
          f"({'+' if gain >= 0 else ''}{gain:.2f} dB vs baseline)")
    if gain < 0.5:
        print("This is not a meaningful improvement over doing nothing yet -- "
              "consider more epochs/samples, a different learning rate, or a "
              "larger model before relying on this checkpoint.")

    if args.export_onnx:
        export_onnx(args.checkpoint, out_path=args.export_onnx, img_size=args.patch_size)


if __name__ == "__main__":
    main()
