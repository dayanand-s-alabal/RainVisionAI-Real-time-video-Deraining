"""
Real-time rain-video deraining + object detection pipeline for autonomous
vehicles, tuned to run on CPU-only machines with 8GB RAM.

Usage:
    python pipeline.py --source path/to/rain_video.mp4 --model ../models/yolov8n.onnx --display
    python pipeline.py --source 0 --model ../models/yolov8n.onnx --output out.mp4

Key CPU/RAM knobs (see --help):
    --derain-width   resolution the derain stage operates at (smaller = faster)
    --det-size       YOLO input size (320/416/640) - 320 is much faster on CPU
    --buffer-size    number of frames averaged for deraining (2-3 is usually enough)
    --no-flow        disable optical-flow alignment for extra speed (less accurate
                      on fast ego-motion, fine for mostly-forward highway driving)
    --detect-every   only run detection every Nth frame (derain still runs every
                      frame; detections are held/re-drawn on skipped frames)
    --threads        onnxruntime intra-op threads, set to your CPU core count
    --adaptive       auto-scale resolution/detection-stride to hold a target FPS
"""
import argparse
import os
import sys

import cv2
import numpy as np

from utils import ThreadedVideoStream, FPSCounter, draw_detections, AdaptiveController
from temporal_derain import TemporalDerain, QUALITY_PRESETS
from yolo_detector import YOLODetector, DRIVING_RELEVANT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=0, help="Video file path or camera index (default: webcam 0)")
    p.add_argument("--model", default=None, help="Path to YOLOv8n .onnx model. If omitted, detection is skipped.")
    p.add_argument("--output", default=None,
                    help="Path to save the CLEAN derained video (no boxes/FPS text burned in) -- "
                         "this is what you want if you just need the deraining result itself")
    p.add_argument("--annotated-output", default=None,
                    help="Path to additionally save a version WITH detection boxes + FPS/debug "
                         "text burned in (for demos/visualization). Can be used together with --output "
                         "to save both in one run.")
    p.add_argument("--display", action="store_true", help="Show a live preview window")
    p.add_argument("--quality", choices=["fast", "balanced", "high"], default="balanced",
                    help="Derain speed/quality preset. 'fast' ~2x the FPS of 'balanced' at lower PSNR "
                         "gain (smaller buffer, no directional streak filter). 'high' pushes PSNR "
                         "further at roughly half the FPS of 'balanced' (larger buffer). Explicit "
                         "--buffer-size/--rain-thresh/etc. flags below override the preset.")
    p.add_argument("--derain-width", type=int, default=480, help="Processing width for deraining stage")
    p.add_argument("--buffer-size", type=int, default=None, help="Frames averaged in temporal derain (>=2); overrides --quality preset")
    p.add_argument("--rain-thresh", type=int, default=None, help="Temporal diff threshold that starts counting as rain (lower = more aggressive removal); overrides --quality preset")
    p.add_argument("--max-valid-flow", type=float, default=18.0, help="Flow magnitude (px) above which a frame is excluded from the median at that pixel (protects real moving objects from ghosting)")
    p.add_argument("--no-flow", action="store_true", help="Disable optical-flow motion compensation")
    p.add_argument("--no-spatial-mask", action="store_true", help="Disable the streak-shape prior, use temporal diff only")
    p.add_argument("--no-chromatic-cue", action="store_true",
                    help="Disable the saturation-drop/brightness-rise rain cue (rain is largely achromatic; "
                         "this helps distinguish it from colored real-object motion). Verified via metrics.py "
                         "to improve PSNR/SSIM; on by default.")
    p.add_argument("--no-adaptive-angle", action="store_true",
                    help="Disable adaptive rain-streak angle estimation (Hough-based) and use the fixed "
                         "near-vertical filter bank instead. NOTE: in testing here this did not show a "
                         "measurable PSNR/SSIM difference either way, even under simulated 40-degree "
                         "wind-blown rain -- it's a correctness improvement (an earlier version had an "
                         "inverted sign bug) more than a proven quality win. Left on by default since it's "
                         "cheap and never measured worse, not because it's confirmed to help.")
    p.add_argument("--det-size", type=int, default=416, help="YOLO input size (320/416/640) - larger catches smaller/farther objects")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold - lower catches more objects but more false positives")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--detect-every", type=int, default=1, help="Run detection every N frames")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="onnxruntime intra-op threads")
    p.add_argument("--driving-classes-only", action="store_true",
                    help="Restrict detection to person/car/truck/bus/bike/traffic-light/stop-sign")
    p.add_argument("--adaptive", action="store_true", help="Auto-adjust resolution/stride to hold target FPS")
    p.add_argument("--debug", action="store_true", help="Print per-frame detection counts and mean rain-mask coverage")
    p.add_argument("--detect-on-raw", action="store_true",
                    help="Run detection on the ORIGINAL (rainy) frame instead of the derained one -- "
                         "useful to isolate whether low detection counts are caused by deraining or by the detector/thresholds")
    p.add_argument("--target-fps", type=float, default=12.0, help="Target FPS for --adaptive controller")
    p.add_argument("--show-derain-only", action="store_true", help="Skip detection, just preview deraining")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    stream = ThreadedVideoStream(source).start()
    w, h = stream.frame_size
    src_fps = stream.fps_hint

    preset = QUALITY_PRESETS[args.quality]
    buffer_size = args.buffer_size if args.buffer_size is not None else preset["buffer_size"]
    rain_thresh = args.rain_thresh if args.rain_thresh is not None else preset["rain_diff_thresh"]

    derainer = TemporalDerain(
        buffer_size=buffer_size,
        use_optical_flow=not args.no_flow,
        resize_width=args.derain_width,
        rain_diff_thresh=rain_thresh,
        max_valid_flow=args.max_valid_flow,
        use_spatial_mask=(not args.no_spatial_mask) and preset["use_spatial_mask"],
        flow_backend=preset["flow_backend"],
        use_chromatic_cue=not args.no_chromatic_cue,
        adaptive_angle=not args.no_adaptive_angle,
    )

    detector = None
    if args.model and not args.show_derain_only:
        keep = DRIVING_RELEVANT if args.driving_classes_only else None
        detector = YOLODetector(
            args.model, conf_thres=args.conf, iou_thres=args.iou, input_size=args.det_size,
            num_threads=args.threads, keep_classes=keep,
        )

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps or 20.0, (w, h))
        if not writer.isOpened():
            sys.exit(f"Could not open output video for writing: {args.output} "
                      f"(check the directory exists and the extension is supported, e.g. .mp4)")

    annotated_writer = None
    if args.annotated_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        annotated_writer = cv2.VideoWriter(args.annotated_output, fourcc, src_fps or 20.0, (w, h))
        if not annotated_writer.isOpened():
            sys.exit(f"Could not open annotated-output video for writing: {args.annotated_output}")

    fps_counter = FPSCounter()
    controller = AdaptiveController(target_fps=args.target_fps) if args.adaptive else None
    last_detections = (np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int))

    frame_idx = 0
    print("Starting pipeline. Press 'q' to quit (if --display is set).")
    try:
        while stream.more():
            ok, frame = stream.read()
            if not ok or frame is None:
                continue

            derain_width = args.derain_width
            det_stride = args.detect_every
            if controller:
                scale, det_stride = controller.update(fps_counter.fps or args.target_fps)
                derain_width = max(160, int(args.derain_width * scale))
                derainer.resize_width = derain_width

            derained = derainer.process(frame)
            detect_input = frame if args.detect_on_raw else derained

            if detector and frame_idx % det_stride == 0:
                last_detections = detector.detect(detect_input)
                if args.debug:
                    boxes, scores, ids = last_detections
                    names = [detector.classes[c] if c < len(detector.classes) else str(c) for c in ids]
                    print(f"[frame {frame_idx}] detections={len(boxes)} "
                          f"classes={names} scores={[round(float(s),2) for s in scores]}")

            fps = fps_counter.update()

            # `derained` stays CLEAN (no boxes/text) for --output. A separate
            # copy gets annotated for --display / --annotated-output, so the
            # saved deraining result is never contaminated with debug overlay
            # or bounding boxes unless explicitly requested.
            need_annotated = args.display or annotated_writer is not None
            if need_annotated:
                display_frame = derained.copy()
                if detector:
                    draw_detections(display_frame, *last_detections, detector.classes)
                cv2.putText(display_frame, f"FPS: {fps:.1f}  derain_w:{derain_width} stride:{det_stride}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                display_frame = None

            if writer:
                writer.write(derained)
            if annotated_writer:
                annotated_writer.write(display_frame)
            if args.display:
                cv2.imshow("Derain + Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        stream.release()
        if writer:
            writer.release()
        if annotated_writer:
            annotated_writer.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_idx} frames. Final smoothed FPS: {fps_counter.fps:.2f}")
        if args.output:
            print(f"Clean derained video saved to: {args.output}")
        if args.annotated_output:
            print(f"Annotated video saved to: {args.annotated_output}")


if __name__ == "__main__":
    main()
