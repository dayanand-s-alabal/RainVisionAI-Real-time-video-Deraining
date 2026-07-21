"""
Shared utilities for the real-time deraining + detection pipeline.
Kept dependency-light (cv2 + numpy only) so it runs on any CPU-only 8GB machine.
"""
import os
import time
import queue
import threading

import cv2
import numpy as np


class ThreadedVideoStream:
    """
    Reads frames on a background thread so that camera/file I/O never blocks
    the compute thread (derain + detection). This alone can give a meaningful
    FPS boost on CPU-only systems where decoding competes with inference.

    Uses a blocking queue.Queue rather than a hand-rolled deque+lock: read()
    blocks on q.get(timeout=...), which properly releases the GIL while
    waiting. An earlier version used a tight busy-wait loop in the consumer
    (spin-checking an empty deque with no backoff), which under Python's GIL
    can starve the reader thread of scheduling time entirely -- the pipeline
    would silently stall after a few frames instead of erroring, especially
    noticeable on machines/thread-schedulers where the busy loop wins the GIL
    tug-of-war. Queue.get()'s internal condition-variable wait avoids that.
    """

    def __init__(self, source, queue_size=8):
        if isinstance(source, str) and not source.isdigit():
            if not os.path.exists(source):
                cwd = os.getcwd()
                raise RuntimeError(
                    f"Video file not found: '{source}' (resolved against current directory "
                    f"'{cwd}').\n"
                    f"  - Check the path is correct relative to where you're running this "
                    f"script from (paths in the README are relative to the src/ folder).\n"
                    f"  - Make sure you're pointing at a real video file you have -- this "
                    f"project doesn't ship a sample video."
                )
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Found '{source}' but OpenCV could not open it as a video "
                f"(unsupported codec/container, or the file is corrupted). "
                f"Try re-encoding it to H.264 mp4, e.g.:\n"
                f"  ffmpeg -i \"{source}\" -c:v libx264 -pix_fmt yuv420p fixed.mp4"
            )
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            try:
                self.q.put(frame, timeout=1.0)
            except queue.Full:
                # Consumer is falling behind (e.g. live camera + slow processing).
                # Drop the oldest queued frame to make room rather than blocking
                # the reader indefinitely -- keeps the stream "live" instead of
                # accumulating ever-growing latency.
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    pass

    def read(self):
        """Blocks briefly waiting for a frame; returns (False, None) once the
        stream has truly ended and the queue is drained."""
        while True:
            try:
                frame = self.q.get(timeout=0.1)
                return True, frame
            except queue.Empty:
                if self.stopped:
                    return False, None
                # still running, just no frame ready yet -- loop and wait again
                # (this blocks on the queue's condition variable, not a spin)

    def more(self):
        return not self.stopped or not self.q.empty()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()

    @property
    def fps_hint(self):
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h


class FPSCounter:
    """Exponential moving average FPS counter (smoother than instantaneous 1/dt)."""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.fps = 0.0
        self._last = None

    def update(self):
        now = time.time()
        if self._last is not None:
            inst_fps = 1.0 / max(now - self._last, 1e-6)
            self.fps = self.alpha * self.fps + (1 - self.alpha) * inst_fps if self.fps > 0 else inst_fps
        self._last = now
        return self.fps


def letterbox(image, new_shape=640, color=(114, 114, 114)):
    """
    Resize + pad image to a square input while preserving aspect ratio.
    Returns the padded image plus the scale ratio and (dw, dh) padding,
    which are needed to map detection boxes back to the original frame.
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = image.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)


def draw_detections(frame, boxes, scores, class_ids, class_names, mask_color=(0, 255, 0)):
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = f"{class_names[cid] if cid < len(class_names) else cid}: {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), mask_color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), mask_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


class AdaptiveController:
    """
    Very small controller that watches recent frame times and suggests
    a processing-resolution scale + detection stride so the pipeline degrades
    gracefully instead of falling further and further behind on weak CPUs.
    """

    def __init__(self, target_fps=15, min_scale=0.5, max_scale=1.0):
        self.target_fps = target_fps
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale = max_scale
        self.detect_stride = 1

    def update(self, current_fps):
        if current_fps < self.target_fps * 0.7:
            self.scale = max(self.min_scale, self.scale - 0.05)
            self.detect_stride = min(4, self.detect_stride + 1)
        elif current_fps > self.target_fps * 1.2:
            self.scale = min(self.max_scale, self.scale + 0.05)
            self.detect_stride = max(1, self.detect_stride - 1)
        return self.scale, self.detect_stride
