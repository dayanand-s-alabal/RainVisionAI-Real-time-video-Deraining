"""
Lightweight object detector wrapper around a YOLOv8-nano ONNX model, run via
ONNX Runtime's CPUExecutionProvider. YOLOv8n (~3.2M params, ~6MB fp32 / ~3MB
int8) is the sweet spot for 8GB RAM / CPU-only real-time use -- it's small
enough to keep RAM usage low and fast enough for several FPS on a modern CPU,
especially after INT8 quantization (see export_and_quantize.py).

Model download (once, requires internet):
    pip install ultralytics
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', opset=12, simplify=True)"
This produces yolov8n.onnx -- place it in models/.
"""
import numpy as np
import cv2
import onnxruntime as ort

from utils import letterbox

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Classes most relevant to driving scenes -- use via `keep_classes` to skip
# scoring/drawing irrelevant categories and speed up NMS/drawing slightly.
DRIVING_RELEVANT = {"person", "bicycle", "car", "motorcycle", "bus", "train",
                     "truck", "traffic light", "stop sign"}


class YOLODetector:
    def __init__(self, model_path, conf_thres=0.4, iou_thres=0.45, input_size=640,
                 num_threads=4, keep_classes=None):
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=so,
                                             providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        # Models exported WITHOUT dynamic=True (the ultralytics default) are
        # baked to a fixed H/W and will throw an ONNXRuntime shape-mismatch
        # error at inference time if you feed them anything else. Rather than
        # let that surface as a confusing crash, read the model's actual
        # expected shape and use it -- overriding whatever --det-size was
        # requested if the model doesn't support dynamic input.
        onnx_shape = self.session.get_inputs()[0].shape  # e.g. [1, 3, 320, 320] or [1, 3, 'height', 'width']
        h_dim, w_dim = onnx_shape[2], onnx_shape[3]
        is_dynamic = isinstance(h_dim, str) or isinstance(w_dim, str)
        if is_dynamic:
            self.input_size = input_size
        else:
            if int(h_dim) != int(w_dim):
                print(f"[YOLODetector] Warning: non-square fixed input {h_dim}x{w_dim}; "
                      f"using height for letterboxing, results may be off.")
            if input_size != int(h_dim):
                print(f"[YOLODetector] Model '{model_path}' was exported with a FIXED "
                      f"input size of {h_dim}x{w_dim} (not dynamic). Overriding "
                      f"requested --det-size {input_size} -> {h_dim}. To use a "
                      f"different size, re-export with dynamic=True (see "
                      f"export_and_quantize.py) or export at the size you want.")
            self.input_size = int(h_dim)

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = COCO_CLASSES
        self.keep_class_ids = (
            {i for i, c in enumerate(COCO_CLASSES) if c in keep_classes}
            if keep_classes else None
        )

    def _preprocess(self, frame):
        img, ratio, (dw, dh) = letterbox(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None, ...]  # NCHW
        return np.ascontiguousarray(img), ratio, (dw, dh)

    def _postprocess(self, output, ratio, pad, orig_shape):
        # YOLOv8 ONNX output: (1, 4 + num_classes, num_anchors)
        pred = np.squeeze(output[0]).T  # -> (num_anchors, 4+num_classes)
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        keep = scores > self.conf_thres
        if self.keep_class_ids is not None:
            keep &= np.isin(class_ids, list(self.keep_class_ids))
        boxes_xywh, scores, class_ids = boxes_xywh[keep], scores[keep], class_ids[keep]
        if len(scores) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        dw, dh = pad
        x = boxes_xywh[:, 0] - dw
        y = boxes_xywh[:, 1] - dh
        x1 = (x - boxes_xywh[:, 2] / 2) / ratio
        y1 = (y - boxes_xywh[:, 3] / 2) / ratio
        x2 = (x + boxes_xywh[:, 2] / 2) / ratio
        y2 = (y + boxes_xywh[:, 3] / 2) / ratio
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        h, w = orig_shape[:2]
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)

        # NMS (cv2 expects x,y,w,h)
        boxes_xywh_for_nms = np.stack([
            boxes_xyxy[:, 0], boxes_xyxy[:, 1],
            boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
        ], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(boxes_xywh_for_nms, scores.tolist(),
                                 self.conf_thres, self.iou_thres)
        idxs = np.array(idxs).flatten() if len(idxs) > 0 else np.array([], dtype=int)

        return boxes_xyxy[idxs], scores[idxs], class_ids[idxs]

    def detect(self, frame):
        inp, ratio, pad = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: inp})
        return self._postprocess(outputs, ratio, pad, frame.shape)
