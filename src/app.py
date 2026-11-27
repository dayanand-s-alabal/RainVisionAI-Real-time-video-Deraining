"""
Web UI backend for RainVisionAI: drag-and-drop video deraining + evaluation metrics.

Run from src/:
    python app.py
Or from project root:
    uvicorn src.app:app --host 127.0.0.1 --port 8000
"""
import csv
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
    import uvicorn
except ImportError as exc:
    raise SystemExit(
        "FastAPI/Uvicorn are required for the web interface.\n"
        "Install them with:\n"
        "    pip install -r requirements-ui.txt"
    ) from exc

from lightweight_derain_net import add_synthetic_rain
from metrics import psnr, ssim, streak_energy, temporal_flicker
from temporal_derain import QUALITY_PRESETS, TemporalDerain
from utils import draw_detections

try:
    from yolo_detector import DRIVING_RELEVANT, YOLODetector
except ImportError:
    DRIVING_RELEVANT = set()
    YOLODetector = None

STATIC_DIR = SRC_DIR / "static"
JOBS_DIR = PROJECT_ROOT / "ui_jobs"
MODELS_DIR = PROJECT_ROOT / "models"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RainVisionAI Web Interface")

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _laplacian_sharpness(gray: np.ndarray) -> float:
    """Variance of Laplacian as a simple focus/sharpness proxy."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _get_ffmpeg_exe() -> Optional[str]:
    """Return path to ffmpeg binary (system PATH or imageio-ffmpeg bundle)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_for_browser(src_path: Path, dst_path: Path) -> bool:
    """Transcode mp4v video to browser-friendly H.264 if ffmpeg is available."""
    ffmpeg = _get_ffmpeg_exe()
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg, "-y", "-i", str(src_path),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dst_path),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        ok = res.returncode == 0 and dst_path.exists() and dst_path.stat().st_size > 0
        if not ok:
            print(f"[warn] ffmpeg transcode failed ({res.returncode}): {res.stderr.decode(errors='ignore')[-300:]}")
        return ok
    except Exception as exc:
        print(f"[warn] ffmpeg transcode error: {exc}")
        return False


def _save_upload(upload: UploadFile, dest: Path) -> None:
    with dest.open("wb") as out_f:
        shutil.copyfileobj(upload.file, out_f)


def _summarize_metrics(rows: list, keys: list) -> Dict[str, Dict[str, float]]:
    summary = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            summary[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }
    return summary


def _write_csv(rows: list, out_csv: Path) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _update_job(job_id: str, **updates: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


def _process_job(
    job_id: str,
    job_dir: Path,
    primary_path: Path,
    clean_path: Optional[Path],
    mode: str,
    quality: str,
    derain_width: int,
    max_frames: int,
    rain_density: float,
    no_flow: bool,
    detect_objects: bool,
    model_name: Optional[str],
) -> None:
    cap_primary = None
    cap_clean = None
    writer_derained = None
    writer_annotated = None
    writer_rainy_preview = None

    try:
        _update_job(job_id, status="processing")

        cap_primary = cv2.VideoCapture(str(primary_path))
        if not cap_primary.isOpened():
            raise RuntimeError("OpenCV could not open the uploaded video. Please try an MP4/AVI/MOV file.")

        if clean_path is not None:
            cap_clean = cv2.VideoCapture(str(clean_path))
            if not cap_clean.isOpened():
                raise RuntimeError("OpenCV could not open the clean ground-truth video.")

        src_fps = cap_primary.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 0 or np.isnan(src_fps):
            src_fps = 24.0
        total_video_frames = int(cap_primary.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if max_frames > 0 and total_video_frames > 0:
            total_frames = min(total_video_frames, max_frames)
        elif max_frames > 0:
            total_frames = max_frames
        else:
            total_frames = total_video_frames

        preset = QUALITY_PRESETS[quality]
        derainer = TemporalDerain(
            buffer_size=preset["buffer_size"],
            use_optical_flow=not no_flow,
            resize_width=derain_width,
            rain_diff_thresh=preset["rain_diff_thresh"],
            use_spatial_mask=preset["use_spatial_mask"],
            flow_backend=preset["flow_backend"],
        )

        detector = None
        if detect_objects and model_name and YOLODetector is not None:
            model_path = (MODELS_DIR / model_name).resolve()
            if model_path.parent == MODELS_DIR.resolve() and model_path.exists():
                detector = YOLODetector(
                    str(model_path),
                    conf_thres=0.25,
                    iou_thres=0.45,
                    input_size=416,
                    num_threads=os.cpu_count() or 4,
                    keep_classes=DRIVING_RELEVANT,
                )

        raw_derained_path = job_dir / "derained_raw.mp4"
        raw_annotated_path = job_dir / "annotated_raw.mp4"
        raw_rainy_preview_path = job_dir / "rainy_input_raw.mp4"

        def _open_writer(path: Path, width: int, height: int) -> cv2.VideoWriter:
            for codec in ("mp4v", "avc1", "XVID"):
                w_out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), src_fps, (width, height))
                if w_out.isOpened():
                    return w_out
            raise RuntimeError(f"Failed to initialize VideoWriter for {path.name}.")

        rows = []
        prev_gray_rainy = None
        prev_gray_derained = None
        detected_class_counts: Dict[str, int] = {}
        n = 0
        t_job_start = time.perf_counter()

        while max_frames <= 0 or n < max_frames:
            ok_p, frame_p = cap_primary.read()
            if not ok_p or frame_p is None:
                break

            fh, fw = frame_p.shape[:2]
            if writer_derained is None:
                writer_derained = _open_writer(raw_derained_path, fw, fh)
                if detector is not None:
                    writer_annotated = _open_writer(raw_annotated_path, fw, fh)
                if mode == "synthetic":
                    writer_rainy_preview = _open_writer(raw_rainy_preview_path, fw, fh)

            if mode == "synthetic":
                clean = frame_p
                rainy = add_synthetic_rain(clean, density=rain_density)
                if writer_rainy_preview is not None:
                    writer_rainy_preview.write(rainy)
            elif mode == "paired":
                ok_c, clean = cap_clean.read()
                if not ok_c or clean is None:
                    break
                if clean.shape[:2] != frame_p.shape[:2]:
                    clean = cv2.resize(clean, (fw, fh))
                rainy = frame_p
            else:
                rainy = frame_p
                clean = None

            t0 = time.perf_counter()
            derained = derainer.process(rainy)
            t1 = time.perf_counter()

            proc_time = t1 - t0
            fps = 1.0 / proc_time if proc_time > 0 else 0.0
            writer_derained.write(derained)

            frame_det_count = None
            if writer_annotated is not None and detector is not None:
                boxes, scores, class_ids = detector.detect(derained)
                frame_det_count = int(len(boxes))
                for cid in class_ids:
                    cname = detector.classes[cid] if cid < len(detector.classes) else str(cid)
                    detected_class_counts[cname] = detected_class_counts.get(cname, 0) + 1
                annotated = derained.copy()
                draw_detections(annotated, boxes, scores, class_ids, detector.classes)
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}  Det: {frame_det_count}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer_annotated.write(annotated)

            gray_rainy = cv2.cvtColor(rainy, cv2.COLOR_BGR2GRAY)
            gray_derained = cv2.cvtColor(derained, cv2.COLOR_BGR2GRAY)

            row: Dict[str, Any] = {
                "frame": n,
                "fps": round(fps, 3),
                "processing_time_ms": round(proc_time * 1000.0, 2),
                "streak_energy_rainy": round(streak_energy(gray_rainy), 5),
                "streak_energy_derained": round(streak_energy(gray_derained), 5),
                "sharpness_rainy": round(_laplacian_sharpness(gray_rainy), 2),
                "sharpness_derained": round(_laplacian_sharpness(gray_derained), 2),
            }
            row["streak_energy_reduction_pct"] = round(
                100.0 * ((row["streak_energy_rainy"] - row["streak_energy_derained"]) / max(row["streak_energy_rainy"], 1e-6)),
                2,
            )
            if prev_gray_rainy is not None:
                row["flicker_rainy"] = round(temporal_flicker(prev_gray_rainy, gray_rainy), 4)
                row["flicker_derained"] = round(temporal_flicker(prev_gray_derained, gray_derained), 4)
            prev_gray_rainy, prev_gray_derained = gray_rainy, gray_derained

            if clean is not None:
                p_rainy = psnr(rainy, clean)
                p_derained = psnr(derained, clean)
                s_rainy = ssim(rainy, clean)
                s_derained = ssim(derained, clean)
                row["psnr_rainy_vs_clean"] = round(p_rainy, 4)
                row["psnr_derained_vs_clean"] = round(p_derained, 4)
                row["psnr_gain"] = round(p_derained - p_rainy, 4)
                row["ssim_rainy_vs_clean"] = round(s_rainy, 4)
                row["ssim_derained_vs_clean"] = round(s_derained, 4)
                row["ssim_gain"] = round(s_derained - s_rainy, 4)

            if frame_det_count is not None:
                row["detections_count"] = frame_det_count

            rows.append(row)
            n += 1

            elapsed = time.perf_counter() - t_job_start
            avg_fps = n / elapsed if elapsed > 0 else fps
            pct = (n / total_frames * 100.0) if total_frames > 0 else 0.0
            eta = ((total_frames - n) / avg_fps) if (total_frames > 0 and avg_fps > 0) else None
            _update_job(
                job_id,
                progress={
                    "processed_frames": n,
                    "total_frames": total_frames,
                    "percent": round(min(pct, 99.0), 1),
                    "current_fps": round(avg_fps, 2),
                    "eta_seconds": round(eta, 1) if eta is not None else None,
                },
            )

        if n == 0:
            raise RuntimeError("No frames could be read from the uploaded video.")

        cap_primary.release()
        cap_primary = None
        if cap_clean is not None:
            cap_clean.release()
            cap_clean = None
        writer_derained.release()
        writer_derained = None
        if writer_annotated is not None:
            writer_annotated.release()
            writer_annotated = None
        if writer_rainy_preview is not None:
            writer_rainy_preview.release()
            writer_rainy_preview = None

        # Transcode to H.264 for reliable HTML5 <video> playback when ffmpeg is available
        final_derained = job_dir / "derained.mp4"
        transcoded = _transcode_for_browser(raw_derained_path, final_derained)
        if not transcoded:
            shutil.move(str(raw_derained_path), str(final_derained))
        else:
            raw_derained_path.unlink(missing_ok=True)

        outputs = {
            "derained_video": f"/api/jobs/{job_id}/files/derained.mp4",
            "mjpeg_derained": f"/api/jobs/{job_id}/mjpeg/derained",
            "mjpeg_input": f"/api/jobs/{job_id}/mjpeg/input",
            "metrics_csv": f"/api/jobs/{job_id}/files/metrics.csv",
            "browser_transcoded": transcoded,
        }

        if mode == "synthetic" and raw_rainy_preview_path.exists():
            final_rainy = job_dir / "rainy_input.mp4"
            if not _transcode_for_browser(raw_rainy_preview_path, final_rainy):
                shutil.move(str(raw_rainy_preview_path), str(final_rainy))
            else:
                raw_rainy_preview_path.unlink(missing_ok=True)
            outputs["input_video"] = f"/api/jobs/{job_id}/files/rainy_input.mp4"
        else:
            final_input = job_dir / "input.mp4"
            if not _transcode_for_browser(primary_path, final_input):
                shutil.copy2(str(primary_path), str(final_input))
            outputs["input_video"] = f"/api/jobs/{job_id}/files/input.mp4"

        if raw_annotated_path.exists():
            final_annotated = job_dir / "annotated.mp4"
            if not _transcode_for_browser(raw_annotated_path, final_annotated):
                shutil.move(str(raw_annotated_path), str(final_annotated))
            else:
                raw_annotated_path.unlink(missing_ok=True)
            outputs["annotated_video"] = f"/api/jobs/{job_id}/files/annotated.mp4"
            outputs["mjpeg_annotated"] = f"/api/jobs/{job_id}/mjpeg/annotated"

        csv_path = job_dir / "metrics.csv"
        _write_csv(rows, csv_path)

        summary_keys = [
            "fps",
            "processing_time_ms",
            "streak_energy_rainy",
            "streak_energy_derained",
            "streak_energy_reduction_pct",
            "flicker_rainy",
            "flicker_derained",
            "sharpness_rainy",
            "sharpness_derained",
        ]
        if mode in ("paired", "synthetic"):
            summary_keys.extend([
                "psnr_rainy_vs_clean",
                "psnr_derained_vs_clean",
                "psnr_gain",
                "ssim_rainy_vs_clean",
                "ssim_derained_vs_clean",
                "ssim_gain",
            ])

        if detector is not None:
            summary_keys.append("detections_count")

        summary_data = _summarize_metrics(rows, summary_keys)
        if detector is not None:
            summary_data["detected_classes"] = detected_class_counts

        _update_job(
            job_id,
            status="completed",
            progress={
                "processed_frames": n,
                "total_frames": n,
                "percent": 100.0,
                "current_fps": round(n / max(time.perf_counter() - t_job_start, 1e-6), 2),
                "eta_seconds": 0.0,
            },
            summary=summary_data,
            per_frame=rows,
            outputs=outputs,
        )
    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc))
    finally:
        for res in (cap_primary, cap_clean, writer_derained, writer_annotated, writer_rainy_preview):
            if res is not None:
                try:
                    res.release()
                except Exception:
                    pass


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI static/index.html not found.")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> Dict[str, Any]:
    models = []
    if MODELS_DIR.exists():
        models = sorted(p.name for p in MODELS_DIR.glob("*.onnx"))
    return {
        "status": "ok",
        "ffmpeg": _get_ffmpeg_exe() is not None,
        "models": models,
        "presets": list(QUALITY_PRESETS.keys()),
    }


@app.post("/api/process")
async def start_process(
    rainy_video: UploadFile = File(...),
    clean_video: Optional[UploadFile] = File(None),
    mode: str = Form("auto"),
    quality: str = Form("balanced"),
    derain_width: int = Form(480),
    max_frames: int = Form(0),
    rain_density: float = Form(0.002),
    no_flow: bool = Form(False),
    detect_objects: bool = Form(False),
    model_name: Optional[str] = Form(None),
) -> Dict[str, Any]:
    if quality not in QUALITY_PRESETS:
        raise HTTPException(status_code=400, detail=f"Invalid quality preset: {quality}")

    resolved_mode = mode
    if mode == "auto":
        resolved_mode = "paired" if clean_video is not None and clean_video.filename else "no-reference"
    elif mode not in ("no-reference", "paired", "synthetic"):
        raise HTTPException(status_code=400, detail=f"Invalid evaluation mode: {mode}")

    if resolved_mode == "paired" and (clean_video is None or not clean_video.filename):
        raise HTTPException(status_code=400, detail="Paired mode requires a clean ground-truth video.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    primary_ext = Path(rainy_video.filename or "input.mp4").suffix or ".mp4"
    primary_path = job_dir / f"upload_primary{primary_ext}"
    _save_upload(rainy_video, primary_path)

    clean_path = None
    if clean_video is not None and clean_video.filename and resolved_mode == "paired":
        clean_ext = Path(clean_video.filename).suffix or ".mp4"
        clean_path = job_dir / f"upload_clean{clean_ext}"
        _save_upload(clean_video, clean_path)

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "mode": resolved_mode,
            "quality": quality,
            "progress": {
                "processed_frames": 0,
                "total_frames": 0,
                "percent": 0.0,
                "current_fps": 0.0,
                "eta_seconds": None,
            },
            "summary": None,
            "per_frame": [],
            "outputs": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_process_job,
        args=(
            job_id,
            job_dir,
            primary_path,
            clean_path,
            resolved_mode,
            quality,
            max(160, min(derain_width, 1920)),
            max(0, max_frames),
            max(0.0002, min(rain_density, 0.02)),
            no_flow,
            detect_objects,
            model_name,
        ),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "mode": resolved_mode}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return dict(job)


@app.get("/api/jobs/{job_id}/files/{filename}")
def get_job_file(job_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = (JOBS_DIR / job_id / safe_name).resolve()
    if not file_path.exists() or file_path.parent != (JOBS_DIR / job_id).resolve():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = "text/csv" if safe_name.endswith(".csv") else "video/mp4"
    return FileResponse(path=str(file_path), media_type=media_type, filename=safe_name)


@app.get("/api/jobs/{job_id}/mjpeg/{which}")
def stream_mjpeg(job_id: str, which: str) -> StreamingResponse:
    """Universal OpenCV MJPEG fallback stream when H.264 transcoding is unavailable."""
    name_map = {
        "input": ("input.mp4", "rainy_input.mp4"),
        "derained": ("derained.mp4",),
        "annotated": ("annotated.mp4",),
    }
    candidates = name_map.get(which)
    if not candidates:
        raise HTTPException(status_code=404, detail="Unknown stream target.")

    job_dir = (JOBS_DIR / job_id).resolve()
    video_path = next((job_dir / c for c in candidates if (job_dir / c).exists()), None)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video not found.")

    def _frame_gen():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / (fps if fps and fps > 0 and not np.isnan(fps) else 20.0)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        break
                ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok_enc:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                    )
                time.sleep(delay)
        finally:
            cap.release()

    return StreamingResponse(_frame_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("Starting RainVisionAI Web UI at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
