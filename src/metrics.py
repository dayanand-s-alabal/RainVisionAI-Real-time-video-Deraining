"""
Evaluation harness for the deraining pipeline: PSNR, SSIM, and supporting
metrics, for reporting derain quality the way deraining papers do.

No extra dependencies beyond what's already in requirements.txt (cv2, numpy)
-- PSNR and SSIM are implemented directly rather than pulled from
scikit-image, so this runs anywhere the main pipeline does.

Three evaluation modes:

1. synthetic  -- the only mode that gives an objective PSNR/SSIM number
   without needing a paired rain/clean dataset. Takes a CLEAN video,
   synthesizes rain onto it frame-by-frame (same generator used in
   lightweight_derain_net.py), runs the synthetic-rain frames through
   TemporalDerain, and reports PSNR/SSIM of (rainy vs clean) and
   (derained vs clean) plus the delta. This is the standard way to get a
   ground-truth number when you don't have a real paired dataset.

   python metrics.py synthetic --clean-video path/to/clean_driving.mp4 --frames 60

2. paired -- if you have an aligned rainy/clean pair (e.g. from a public
   deraining video dataset such as NTURain or RainSynAll100, where every
   rainy frame has a corresponding clean ground-truth frame), this runs
   derain on the rainy video and compares directly against the clean video.

   python metrics.py paired --rainy-video rain.mp4 --clean-video gt.mp4

3. no-reference -- for real rain footage with no ground truth available.
   Reports a self-referential streak-energy reduction score (using the same
   directional streak filter the derain algorithm itself uses) and a
   temporal-flicker consistency score, comparing the original vs derained
   video. There's no ground truth here so there's no PSNR/SSIM -- these are
   proxy signals, not absolute quality numbers.

   python metrics.py no-reference --rainy-video real_footage.mp4

All modes write a per-frame CSV to --out-csv and print a summary table.
"""
import argparse
import csv
import sys
import time

import cv2
import numpy as np

from temporal_derain import TemporalDerain, directional_streak_response, QUALITY_PRESETS
from lightweight_derain_net import add_synthetic_rain


# --------------------------------------------------------------------------
# Core metrics (self-contained, no scikit-image dependency)
# --------------------------------------------------------------------------

def psnr(img1, img2, max_val=255.0):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse <= 1e-10:
        return 100.0
    return 10.0 * np.log10((max_val ** 2) / mse)


def _ssim_single_channel(img1, img2):
    """Standard single-scale SSIM (Wang et al. 2004), 11x11 Gaussian window."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def ssim(img1, img2):
    if img1.ndim == 3:
        return float(np.mean([_ssim_single_channel(img1[..., c], img2[..., c]) for c in range(img1.shape[2])]))
    return _ssim_single_channel(img1, img2)


def streak_energy(gray):
    """Sum of the directional streak-matched-filter response -- a proxy for
    'how much rain-streak-like structure is present'. Lower is better."""
    return float(directional_streak_response(gray).mean())


def temporal_flicker(prev_gray, curr_gray):
    """Simple no-reference temporal-consistency proxy: mean absolute frame
    difference. High values on a derained video vs the original can indicate
    the derain process is introducing flicker rather than removing rain."""
    return float(np.mean(cv2.absdiff(prev_gray, curr_gray)))


# --------------------------------------------------------------------------
# Evaluation modes
# --------------------------------------------------------------------------

def _build_derainer(args):
    preset = QUALITY_PRESETS[args.quality]
    buffer_size = args.buffer_size if args.buffer_size is not None else preset["buffer_size"]
    rain_thresh = args.rain_thresh if args.rain_thresh is not None else preset["rain_diff_thresh"]
    return TemporalDerain(
        buffer_size=buffer_size,
        use_optical_flow=not args.no_flow,
        resize_width=args.derain_width,
        rain_diff_thresh=rain_thresh,
        use_spatial_mask=preset["use_spatial_mask"],
        flow_backend=preset["flow_backend"],
        use_chromatic_cue=not args.no_chromatic_cue,
        adaptive_angle=not args.no_adaptive_angle,
    )


def _print_summary(rows, keys):
    print("\n=== Summary (mean over all evaluated frames) ===")
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            print(f"  {k}: {np.mean(vals):.3f}  (min {np.min(vals):.3f}, max {np.max(vals):.3f})")


def _write_csv(rows, out_csv):
    if not rows:
        return
    # Union of keys across all rows, in first-seen order -- don't assume the
    # first row has every field (e.g. no-reference mode's flicker metrics
    # are undefined on frame 0, since they need a previous frame).
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-frame metrics written to {out_csv}")


def eval_synthetic(args):
    cap = cv2.VideoCapture(args.clean_video)
    if not cap.isOpened():
        sys.exit(f"Could not open clean video: {args.clean_video}")

    derainer = _build_derainer(args)

    rows = []
    n = 0
    while args.frames <= 0 or n < args.frames:
        ok, clean = cap.read()
        if not ok:
            break
        rainy = add_synthetic_rain(clean, density=args.rain_density)
        start_time = time.perf_counter()
        derained = derainer.process(rainy)
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        row = {
            "frame": n,
            "fps": fps,
            "processing_time_ms": processing_time * 1000,
            "psnr_rainy_vs_clean": psnr(rainy, clean),
            "psnr_derained_vs_clean": psnr(derained, clean),
            "ssim_rainy_vs_clean": ssim(rainy, clean),
            "ssim_derained_vs_clean": ssim(derained, clean),
        }
        row["psnr_gain"] = row["psnr_derained_vs_clean"] - row["psnr_rainy_vs_clean"]
        row["ssim_gain"] = row["ssim_derained_vs_clean"] - row["ssim_rainy_vs_clean"]
        rows.append(row)
        n += 1
        if args.verbose:
            print(f"frame {n}: PSNR {row['psnr_rainy_vs_clean']:.2f} -> {row['psnr_derained_vs_clean']:.2f} dB "
                  f"| SSIM {row['ssim_rainy_vs_clean']:.3f} -> {row['ssim_derained_vs_clean']:.3f}")

    cap.release()
    _print_summary(rows, ["fps", "processing_time_ms", "psnr_rainy_vs_clean", "psnr_derained_vs_clean", "psnr_gain",
                           "ssim_rainy_vs_clean", "ssim_derained_vs_clean", "ssim_gain"])
    _write_csv(rows, args.out_csv)


def eval_paired(args):
    cap_r = cv2.VideoCapture(args.rainy_video)
    cap_c = cv2.VideoCapture(args.clean_video)
    if not cap_r.isOpened() or not cap_c.isOpened():
        sys.exit("Could not open rainy_video and/or clean_video.")

    derainer = _build_derainer(args)

    rows = []
    n = 0
    while args.frames <= 0 or n < args.frames:
        ok_r, rainy = cap_r.read()
        ok_c, clean = cap_c.read()
        if not ok_r or not ok_c:
            break
        if rainy.shape != clean.shape:
            clean = cv2.resize(clean, (rainy.shape[1], rainy.shape[0]))
        start_time = time.perf_counter()
        derained = derainer.process(rainy)
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        row = {
            "frame": n,
            "fps": fps,
            "processing_time_ms": processing_time * 1000,
            "psnr_rainy_vs_clean": psnr(rainy, clean),
            "psnr_derained_vs_clean": psnr(derained, clean),
            "ssim_rainy_vs_clean": ssim(rainy, clean),
            "ssim_derained_vs_clean": ssim(derained, clean),
        }
        row["psnr_gain"] = row["psnr_derained_vs_clean"] - row["psnr_rainy_vs_clean"]
        row["ssim_gain"] = row["ssim_derained_vs_clean"] - row["ssim_rainy_vs_clean"]
        rows.append(row)
        n += 1
        if args.verbose:
            print(f"frame {n}: PSNR {row['psnr_rainy_vs_clean']:.2f} -> {row['psnr_derained_vs_clean']:.2f} dB "
                  f"| SSIM {row['ssim_rainy_vs_clean']:.3f} -> {row['ssim_derained_vs_clean']:.3f}")

    cap_r.release()
    cap_c.release()
    _print_summary(rows, ["fps", "processing_time_ms", "psnr_rainy_vs_clean", "psnr_derained_vs_clean", "psnr_gain",
                           "ssim_rainy_vs_clean", "ssim_derained_vs_clean", "ssim_gain"])
    _write_csv(rows, args.out_csv)


def eval_no_reference(args):
    cap = cv2.VideoCapture(args.rainy_video)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.rainy_video}")

    derainer = _build_derainer(args)

    rows = []
    prev_gray_rainy, prev_gray_derained = None, None
    n = 0
    while args.frames <= 0 or n < args.frames:
        ok, rainy = cap.read()
        if not ok:
            break
        start_time = time.perf_counter()
        derained = derainer.process(rainy)
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        gray_rainy = cv2.cvtColor(rainy, cv2.COLOR_BGR2GRAY)
        gray_derained = cv2.cvtColor(derained, cv2.COLOR_BGR2GRAY)

        row = {
            "frame": n,
            "fps": fps,
            "processing_time_ms": processing_time * 1000,
            "streak_energy_rainy": streak_energy(gray_rainy),
            "streak_energy_derained": streak_energy(gray_derained),
        }
        row["streak_energy_reduction_pct"] = 100.0 * (
            (row["streak_energy_rainy"] - row["streak_energy_derained"]) /
            max(row["streak_energy_rainy"], 1e-6)
        )
        if prev_gray_rainy is not None:
            row["flicker_rainy"] = temporal_flicker(prev_gray_rainy, gray_rainy)
            row["flicker_derained"] = temporal_flicker(prev_gray_derained, gray_derained)
        rows.append(row)
        prev_gray_rainy, prev_gray_derained = gray_rainy, gray_derained
        n += 1
        if args.verbose:
            print(f"frame {n}: streak energy {row['streak_energy_rainy']:.4f} -> "
                  f"{row['streak_energy_derained']:.4f} "
                  f"({row['streak_energy_reduction_pct']:.1f}% reduction)")

    cap.release()
    _print_summary(rows, ["fps", "processing_time_ms", "streak_energy_rainy", "streak_energy_derained",
                           "streak_energy_reduction_pct", "flicker_rainy", "flicker_derained"])
    print("\nNote: no ground truth is available in this mode, so there is no PSNR/SSIM here --"
          " streak-energy reduction and flicker are proxy signals, not absolute quality scores."
          " For a real PSNR/SSIM number, use `synthetic` mode on a clean clip of similar footage,"
          " or `paired` mode if you have an aligned rainy/clean dataset.")
    _write_csv(rows, args.out_csv)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--quality", choices=["fast", "balanced", "high"], default="balanced",
                         help="Derain preset (see pipeline.py --quality). Explicit --buffer-size/--rain-thresh override it.")
    common.add_argument("--buffer-size", type=int, default=None)
    common.add_argument("--derain-width", type=int, default=480)
    common.add_argument("--rain-thresh", type=int, default=None)
    common.add_argument("--no-flow", action="store_true")
    common.add_argument("--no-chromatic-cue", action="store_true")
    common.add_argument("--no-adaptive-angle", action="store_true")
    common.add_argument("--frames", type=int, default=60, help="Max frames to evaluate (<=0 = all)")
    common.add_argument("--out-csv", default="metrics_output.csv")
    common.add_argument("--verbose", action="store_true")

    p_syn = sub.add_parser("synthetic", parents=[common], help="Synthesize rain on a clean video, measure PSNR/SSIM")
    p_syn.add_argument("--clean-video", required=True)
    p_syn.add_argument("--rain-density", type=float, default=0.002,
                        help="Synthetic rain density. 0.0015=light, 0.002=moderate, 0.003+=heavy (see README benchmark table)")
    p_syn.set_defaults(func=eval_synthetic)

    p_pair = sub.add_parser("paired", parents=[common], help="Compare derained rainy video against aligned clean ground truth")
    p_pair.add_argument("--rainy-video", required=True)
    p_pair.add_argument("--clean-video", required=True)
    p_pair.set_defaults(func=eval_paired)

    p_nr = sub.add_parser("no-reference", parents=[common], help="No ground truth: streak-energy + flicker proxy metrics")
    p_nr.add_argument("--rainy-video", required=True)
    p_nr.set_defaults(func=eval_no_reference)

    return p.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
