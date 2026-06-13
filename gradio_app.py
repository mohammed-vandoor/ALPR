import os
import cv2
import time
import torch
import gradio as gr
from PIL import Image
from collections import Counter
from fast_alpr import ALPR
from torchvision import transforms
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import hf_hub_download
from fast_vehicle_classifier import detect_color_hsv, detect_color_with_conf

# ── Constants ─────────────────────────────────────────────────────────────────
VEHICLE_CLASS_IDS = [2, 5, 7]
YOLO_PATH         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolo11n.pt")
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_VEHICLE_AREA = 0.04
MIN_CONFIDENCE   = 0.55
CLASSIFY_EVERY_N = 3
PLATE_EVERY_N    = 4
MAX_VIDEO_WIDTH  = 1280

CAR_BRANDS = [
    "Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes-Benz", "Volkswagen",
    "Audi", "Nissan", "Hyundai", "Kia", "Mazda", "Subaru", "Lexus", "Jeep",
    "Ram", "GMC", "Cadillac", "Volvo", "Porsche", "Land Rover", "Range Rover",
    "Jaguar", "Mitsubishi", "Suzuki", "Renault", "Peugeot", "Citroën", "Fiat",
    "Tesla", "Chrysler", "Dodge", "Buick", "Infiniti", "Acura", "Genesis",
]
_BRAND_PROMPTS = [f"a photo of a {b} car" for b in CAR_BRANDS]

# ── Model loading (done once at startup) ──────────────────────────────────────
print(f"Loading models on {DEVICE}...")

_alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-s-v1-global-model",
)

_detector = YOLO(YOLO_PATH)

_clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_text_inputs    = _clip_processor(text=_BRAND_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    _text_feats = _clip_model.text_model(**_text_inputs)
    _text_feats = _clip_model.text_projection(_text_feats.pooler_output)
    _text_feats = (_text_feats / _text_feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)).detach()

print("All models loaded.")


# ── Helper functions ──────────────────────────────────────────────────────────
def classify_crop(crop_bgr):
    color, color_conf = detect_color_with_conf(crop_bgr)
    pil   = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
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


def track_vehicles(frame_bgr):
    fh, fw = frame_bgr.shape[:2]
    frame_area = fh * fw
    results = _detector.track(frame_bgr, persist=True, tracker="bytetrack.yaml",
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
        x1c, y1c = max(0, x1), max(0, y1)
        crop = frame_bgr[y1c:y2, x1c:x2]
        if crop.size == 0:
            continue
        tracks.append({"track_id": int(tid.item()), "bbox": (x1, y1, x2, y2),
                        "crop": crop, "confidence": conf.item(), "norm_area": norm_area})
    return tracks


def process_plates(frame_rgb):
    results = []
    try:
        detections = _alpr.run(frame_rgb)
        for d in detections:
            if d.ocr and d.ocr[0].text:
                bb = d.detection.bounding_box
                results.append({
                    "plate": d.ocr[0].text.upper(),
                    "confidence": d.ocr[0].confidence,
                    "bbox": (int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)),
                })
    except Exception:
        pass
    return results


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
        cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
        vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
        dist = ((cx - vcx) ** 2 + (cy - vcy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best = dist, p
    return best


def draw_frame(frame_rgb, tracks, plate_results, track_store):
    img = frame_rgb.copy()
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid = t["track_id"]
        info = track_store.get(tid, {})
        label = f"#{tid} {info.get('brand','?')} {info.get('color','?')} {info.get('conf',0):.0%}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    for p in plate_results:
        px1, py1, px2, py2 = p["bbox"]
        cv2.rectangle(img, (px1, py1), (px2, py2), (255, 140, 0), 2)
        cv2.putText(img, p["plate"], (px1, max(py1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)
    return img


# ── Main processing generator ─────────────────────────────────────────────────
def process_video(video_path):
    if video_path is None:
        yield None, []
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = min(1.0, MAX_VIDEO_WIDTH / raw_w) if raw_w > MAX_VIDEO_WIDTH else 1.0

    frame_idx            = 0
    plate_counter        = 0
    last_plate           = []
    track_store          = {}
    track_classify_count = {}
    log_rows             = []
    row_count            = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if scale < 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        tracks    = track_vehicles(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plate_counter += 1
        if plate_counter % PLATE_EVERY_N == 0:
            last_plate = process_plates(frame_rgb)

        did_classify = False
        for t in tracks:
            tid  = t["track_id"]
            crop = t["crop"]
            track_classify_count[tid] = track_classify_count.get(tid, 0) + 1
            if track_classify_count[tid] % CLASSIFY_EVERY_N != 1:
                continue

            color, color_conf, brand, brand_conf = classify_crop(crop)
            did_classify = True

            matched = match_plate(last_plate, t["bbox"])
            plate_text = matched["plate"] if matched else "—"

            track_store[tid] = {"color": color, "brand": brand, "conf": brand_conf}

            row_count += 1
            timestamp = frame_idx / fps
            mins = int(timestamp) // 60
            secs = int(timestamp) % 60
            log_rows.append([
                tid,
                plate_text,
                color,
                f"{color_conf:.0%}",
                brand,
                f"{brand_conf:.0%}",
                f"{mins:02d}:{secs:02d}",
            ])

        # Yield annotated frame + updated table
        if frame_idx % 5 == 0 or did_classify:
            annotated = draw_frame(frame_rgb, tracks, last_plate, track_store)
            yield annotated, log_rows

    cap.release()
    yield annotated if 'annotated' in dir() else None, log_rows


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="ALPR — Vehicle Identification", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Vehicle Identification — ALPR")
    gr.Markdown("Upload a video to detect vehicles, predict brand, colour and licence plate.")

    with gr.Row():
        with gr.Column(scale=3):
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            run_btn     = gr.Button("▶️ Process Video", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### Device")
            gr.Markdown(f"Running on: **{DEVICE}**")

    with gr.Row():
        with gr.Column(scale=3):
            frame_out = gr.Image(label="Live Detection Feed", streaming=True)
        with gr.Column(scale=2):
            table_out = gr.Dataframe(
                headers=["Vehicle ID", "Plate", "Colour", "Colour Conf", "Brand", "Brand Conf", "Time"],
                label="Detections",
                wrap=True,
            )

    run_btn.click(
        fn=process_video,
        inputs=[video_input],
        outputs=[frame_out, table_out],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
