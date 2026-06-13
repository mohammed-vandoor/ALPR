import os
import cv2
import time
import torch
import queue
import threading
import asyncio
import base64
from typing import List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
from ultralytics import YOLO
from fast_alpr import ALPR
from transformers import CLIPProcessor, CLIPModel
from fast_vehicle_classifier import detect_color_with_conf

# ── Constants ──────────────────────────────────────────────────────────────────
VEHICLE_CLASS_IDS = [2, 5, 7]
YOLO_PATH         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolo11n.pt")
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_VEHICLE_AREA  = 0.04
MIN_CONFIDENCE    = 0.55
CLASSIFY_EVERY_N  = 3
PLATE_EVERY_N     = 4
MAX_VIDEO_WIDTH   = 1280
JPEG_QUALITY      = 70   # lower = faster streaming

CAR_BRANDS = [
    "Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes-Benz", "Volkswagen",
    "Audi", "Nissan", "Hyundai", "Kia", "Mazda", "Subaru", "Lexus", "Jeep",
    "Ram", "GMC", "Cadillac", "Volvo", "Porsche", "Land Rover", "Range Rover",
    "Jaguar", "Mitsubishi", "Suzuki", "Renault", "Peugeot", "Citroën", "Fiat",
    "Tesla", "Chrysler", "Dodge", "Buick", "Infiniti", "Acura", "Genesis",
]
_BRAND_PROMPTS = [f"a photo of a {b} car" for b in CAR_BRANDS]

# ── Load models once at startup ────────────────────────────────────────────────
print(f"[server] Loading models on {DEVICE}...")

_alpr      = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end",
                  ocr_model="cct-s-v1-global-model")
_detector  = YOLO(YOLO_PATH)

_clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_text_inputs    = _clip_processor(text=_BRAND_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    _tf = _clip_model.text_model(**_text_inputs)
    _tf = _clip_model.text_projection(_tf.pooler_output)
    _text_feats = (_tf / _tf.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)).detach()

print("[server] Models loaded.")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# ── Shared state (one processing job at a time) ────────────────────────────────
_job_lock    = threading.Lock()
_log_rows: List[Dict[str, Any]] = []
_log_lock    = threading.Lock()
_active_job  = False

# ── Helper functions ───────────────────────────────────────────────────────────
def classify_crop(crop_bgr):
    color, color_conf = detect_color_with_conf(crop_bgr)
    pil        = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    img_inputs = _clip_processor(images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        img_out   = _clip_model.vision_model(**img_inputs)
        img_feats = _clip_model.visual_projection(img_out.pooler_output)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        logits    = (img_feats @ _text_feats.T) * _clip_model.logit_scale.exp()
        probs     = torch.softmax(logits[0], dim=0)
    best_idx   = probs.argmax().item()
    brand_conf = probs[best_idx].item()
    return color, color_conf, CAR_BRANDS[best_idx], brand_conf


def process_plates(frame_rgb):
    results = []
    try:
        detections = _alpr.predict(frame_rgb)
        for r in (detections or []):
            conf = r.ocr.confidence if r.ocr else 0.0
            if isinstance(conf, list):
                conf = conf[0] if conf else 0.0
            bb   = r.detection.bounding_box if r.detection else None
            bbox = (int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)) if bb else None
            text = r.ocr.text.upper() if r.ocr and r.ocr.text else None
            if text and bbox:
                results.append({"plate": text, "confidence": float(conf), "bbox": bbox})
    except Exception:
        pass
    return results


def track_vehicles(frame_bgr):
    fh, fw    = frame_bgr.shape[:2]
    frame_area = fh * fw
    results   = _detector.track(frame_bgr, persist=True, tracker="bytetrack.yaml",
                                 classes=VEHICLE_CLASS_IDS, verbose=False)
    tracks = []
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return tracks
    for box, tid, conf in zip(results[0].boxes.xyxy,
                               results[0].boxes.id,
                               results[0].boxes.conf):
        if conf.item() < MIN_CONFIDENCE:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        norm_area = ((x2 - x1) * (y2 - y1)) / frame_area
        if norm_area < MIN_VEHICLE_AREA:
            continue
        crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        tracks.append({"track_id": int(tid.item()), "bbox": (x1, y1, x2, y2),
                        "crop": crop, "confidence": conf.item()})
    return tracks


def match_plate(plate_results, vehicle_bbox):
    if not plate_results or vehicle_bbox is None:
        return None
    vx1, vy1, vx2, vy2 = vehicle_bbox
    best, best_dist = None, float("inf")
    for p in plate_results:
        px1, py1, px2, py2 = p["bbox"]
        ox = max(0, min(vx2, px2) - max(vx1, px1))
        oy = max(0, min(vy2, py2) - max(vy1, py1))
        if ox * oy > 0:
            return p
        cx, cy   = (px1 + px2) / 2, (py1 + py2) / 2
        vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
        dist = ((cx - vcx) ** 2 + (cy - vcy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best = dist, p
    return best


def draw_frame(frame_rgb, tracks, plate_results, track_store):
    img = frame_rgb.copy()
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid  = t["track_id"]
        info = track_store.get(tid, {})
        label = f"#{tid} {info.get('brand','?')}  {info.get('color','?')}  {info.get('conf',0):.0%}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), (0, 200, 0), -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    for p in plate_results:
        px1, py1, px2, py2 = p["bbox"]
        cv2.rectangle(img, (px1, py1), (px2, py2), (255, 140, 0), 2)
        cv2.putText(img, p["plate"], (px1, max(py1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 140, 0), 2)
    return img


def frame_to_jpeg_b64(frame_rgb) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path) as f:
        return f.read()


@app.get("/log")
async def get_log():
    with _log_lock:
        return JSONResponse({"rows": list(_log_rows)})


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    global _active_job
    with _job_lock:
        if _active_job:
            return JSONResponse({"error": "A video is already being processed."}, status_code=409)
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    return JSONResponse({"path": tmp_path})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _active_job
    await ws.accept()

    try:
        data = await ws.receive_json()
        video_path = data.get("path")
        if not video_path or not os.path.exists(video_path):
            await ws.send_json({"error": "Invalid video path"})
            await ws.close()
            return
    except Exception:
        await ws.close()
        return

    with _job_lock:
        if _active_job:
            await ws.send_json({"error": "Busy"})
            await ws.close()
            return
        _active_job = True

    with _log_lock:
        _log_rows.clear()

    try:
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        scale = min(1.0, MAX_VIDEO_WIDTH / raw_w) if raw_w > MAX_VIDEO_WIDTH else 1.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        track_store   = {}
        last_plate    = []
        state_lock    = threading.Lock()
        infer_q       = queue.Queue(maxsize=4)
        stop_evt      = threading.Event()
        row_count_ref = [0]

        def inference_worker():
            tcc = {}
            while not stop_evt.is_set():
                try:
                    tid, crop, bbox, ts = infer_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                tcc[tid] = tcc.get(tid, 0) + 1
                if tcc[tid] % CLASSIFY_EVERY_N != 1:
                    infer_q.task_done()
                    continue
                color, color_conf, brand, brand_conf = classify_crop(crop)
                mins, secs = int(ts) // 60, int(ts) % 60
                with state_lock:
                    track_store[tid] = {"color": color, "color_conf": color_conf,
                                        "brand": brand, "conf": brand_conf}
                    matched   = match_plate(last_plate, bbox)
                    plate_txt = matched["plate"] if matched else "—"
                    row_count_ref[0] += 1
                    _log_rows.append({
                        "id":          tid,
                        "plate":       plate_txt,
                        "color":       color,
                        "color_conf":  f"{color_conf:.0%}",
                        "brand":       brand,
                        "brand_conf":  f"{brand_conf:.0%}",
                        "time":        f"{mins:02d}:{secs:02d}",
                    })
                infer_q.task_done()

        worker = threading.Thread(target=inference_worker, daemon=True)
        worker.start()

        frame_idx     = 0
        plate_counter = 0
        loop          = asyncio.get_event_loop()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if scale < 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            timestamp = frame_idx / fps
            tracks    = track_vehicles(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            plate_counter += 1
            if plate_counter % PLATE_EVERY_N == 0:
                new_plates = process_plates(frame_rgb)
                with state_lock:
                    if new_plates:
                        last_plate.clear()
                        last_plate.extend(new_plates)

            for t in tracks:
                try:
                    infer_q.put_nowait((t["track_id"], t["crop"], t["bbox"], timestamp))
                except queue.Full:
                    pass

            # Send annotated frame every 3rd frame as base64 JPEG over WebSocket
            if frame_idx % 3 == 0:
                with state_lock:
                    plates_snap = list(last_plate)
                    store_snap  = dict(track_store)
                annotated = draw_frame(frame_rgb, tracks, plates_snap, store_snap)
                b64       = frame_to_jpeg_b64(annotated)
                progress  = round(frame_idx / total * 100)
                await ws.send_json({"frame": b64, "progress": progress,
                                    "rows": len(_log_rows)})

        stop_evt.set()
        worker.join(timeout=5)
        cap.release()

        with state_lock:
            rows_snap = list(_log_rows)
        await ws.send_json({"done": True, "rows": len(rows_snap)})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        with _job_lock:
            _active_job = False
        stop_evt.set()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
