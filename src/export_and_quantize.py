"""
One-time setup helper:
  1. Exports yolov8n.pt -> yolov8n.onnx (requires `pip install ultralytics` once,
     not needed at pipeline runtime).
  2. Dynamically quantizes the ONNX model to INT8, which roughly halves model
     size and typically gives a 1.5-3x CPU speedup with a small accuracy drop --
     a good trade-off on 8GB RAM / CPU-only hardware.

By default this exports with dynamic=True, meaning the resulting ONNX model
accepts any input resolution (so pipeline.py's --det-size can be changed
freely without re-exporting). Use --fixed if you specifically want a
static-shape export (occasionally marginally faster, but then --det-size
in pipeline.py MUST match --imgsz exactly or you'll get an ONNXRuntime
shape-mismatch error -- yolo_detector.py auto-detects and warns about this
either way, but dynamic export avoids the foot-gun entirely).

Run:
    pip install ultralytics onnxruntime
    python export_and_quantize.py --out-dir ../models --imgsz 416
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../models")
    ap.add_argument("--imgsz", type=int, default=416,
                     help="Export input size (or the default size for a dynamic export). "
                          "320=fastest, 416=balanced, 640=most accurate/slowest on CPU.")
    ap.add_argument("--fixed", action="store_true",
                     help="Export with a FIXED input size instead of dynamic. Then "
                          "pipeline.py's --det-size must exactly equal --imgsz.")
    ap.add_argument("--skip-export", action="store_true", help="Skip .pt->.onnx export, quantize existing onnx")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    onnx_path = os.path.join(args.out_dir, "yolov8n.onnx")

    if not args.skip_export:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # auto-downloads pretrained COCO weights
        exported = model.export(format="onnx", opset=12, simplify=True,
                                 imgsz=args.imgsz, dynamic=not args.fixed)
        if os.path.abspath(exported) != os.path.abspath(onnx_path):
            os.replace(exported, onnx_path)
        shape_desc = f"fixed {args.imgsz}x{args.imgsz}" if args.fixed else "dynamic (any size)"
        print(f"Exported ONNX model to {onnx_path} [{shape_desc}]")
        if args.fixed:
            print(f"NOTE: since this is a fixed-shape export, always run pipeline.py "
                  f"with --det-size {args.imgsz} to match.")

    from onnxruntime.quantization import quantize_dynamic, QuantType
    quant_path = os.path.join(args.out_dir, "yolov8n_int8.onnx")
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)
    print(f"Quantized INT8 model written to {quant_path}")
    print("Use --model pointing at the int8 file in pipeline.py for the fastest CPU inference.")


if __name__ == "__main__":
    main()
