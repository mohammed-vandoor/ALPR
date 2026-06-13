import os
import cv2
import time
import torch
import timm
import streamlit as st
from collections import deque, Counter
from fast_alpr import ALPR
from PIL import Image
import numpy as np
from torchvision import transforms
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from fast_vehicle_classifier import detect_color_hsv

VEHICLE_CLASS_IDS = [2, 5, 7]
YOLO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolo11n.pt")

# ── Device: use CUDA (Colab GPU) if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Tracking settings
MIN_VEHICLE_AREA   = 0.04   # vehicle bbox must be >= 4% of frame area to classify
MIN_CONFIDENCE     = 0.55   # minimum YOLO box confidence to count a frame
FRAMES_TO_LOCK     = 6      # good frames needed before finalising a vehicle result
TRACK_LOST_FRAMES  = 10     # frames without a track before we consider it gone
PLATE_EVERY_N      = 4      # run plate detection every N frames (expensive)
CLASSIFY_EVERY_N   = 3      # only run brand/colour on every Nth frame per track
MAX_VIDEO_WIDTH    = 1280   # downscale wide videos to this width before processing

# ── Brand classifier transform (defined once, reused per frame)
_BRAND_TFM = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def most_common(values):
    """Return the most frequently seen value in a list, ignoring blanks."""
    clean = [v for v in values if v and v not in ("Unknown", "—", "")]
    if not clean:
        return "—"
    return Counter(clean).most_common(1)[0][0]


def classify_crop(crop_bgr, jordo_model, class_mapping):
    """Run colour + brand classification on a single vehicle crop.
    Returns (color, brand, brand_confidence). Runs on GPU if available.
    """
    color = detect_color_hsv(crop_bgr)
    pil   = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = _BRAND_TFM(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(jordo_model(tensor), dim=1)[0]
    brand_conf = probs.max().item()
    idx        = probs.argmax().item()
    brand      = extract_brand(class_mapping[idx])
    return color, brand, brand_conf


@st.cache_resource
def load_alpr():
    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-s-v1-global-model",
    )


@st.cache_resource
def load_jordo23():
    path = hf_hub_download(repo_id="Jordo23/vehicle-classifier", filename="vehicle_classifier.pth")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=8949)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    return model, checkpoint["class_mapping"]


@st.cache_resource
def load_yolo():
    return YOLO(YOLO_PATH)


def extract_brand(label):
    two_word_brands = ["Range Rover", "Land Rover", "Mercedes-Benz", "Alfa Romeo", "Aston Martin", "Rolls-Royce"]
    for brand in two_word_brands:
        if label.startswith(brand):
            return brand
    return label.split()[0] if label else label


def track_vehicles(frame_bgr, detector):
    """Run ByteTrack on a single frame. Returns list of tracked vehicles:
    each with track_id, bbox, crop, confidence, norm_area.
    Only returns vehicles large enough to classify reliably.
    """
    fh, fw = frame_bgr.shape[:2]
    results = detector.track(frame_bgr, persist=True, tracker="bytetrack.yaml",
                             classes=VEHICLE_CLASS_IDS, verbose=False)
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return []

    boxes      = results[0].boxes.xyxy.cpu().numpy()
    track_ids  = results[0].boxes.id.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()

    tracked = []
    for box, tid, conf in zip(boxes, track_ids, confidences):
        if conf < MIN_CONFIDENCE:
            continue
        x1, y1, x2, y2 = map(int, box)
        norm_area = (x2 - x1) * (y2 - y1) / (fw * fh)
        if norm_area < MIN_VEHICLE_AREA:
            continue
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        tracked.append({"track_id": tid, "bbox": (x1, y1, x2, y2),
                        "crop": crop, "confidence": float(conf),
                        "norm_area": norm_area})
    return tracked


def process_plates(image_array, alpr):
    """Read all licence plates in the frame."""
    results = alpr.predict(image_array)
    out = []
    for r in (results or []):
        conf = r.ocr.confidence if r.ocr else 0.0
        if conf is None: conf = 0.0
        elif isinstance(conf, list): conf = conf[0] if conf else 0.0
        bb   = r.detection.bounding_box if r.detection else None
        bbox = (bb.x1, bb.y1, bb.x2, bb.y2) if bb else None
        out.append({"plate": r.ocr.text if r.ocr else "UNKNOWN",
                    "confidence": float(conf), "bbox": bbox})
    return out


def match_plate_to_vehicle(plate_results, vehicle):
    """Return the plate that belongs to the primary vehicle (overlap first, then proximity)."""
    if not plate_results or vehicle is None:
        return None
    vx1, vy1, vx2, vy2 = vehicle["bbox"]
    best, best_score = None, -1
    for p in plate_results:
        if p["bbox"] is None:
            continue
        px1, py1, px2, py2 = [int(v) for v in p["bbox"]]
        ox = max(0, min(px2, vx2) - max(px1, vx1))
        oy = max(0, min(py2, vy2) - max(py1, vy1))
        if ox * oy > 0:
            return p
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
        score = -(((pcx - vcx) ** 2 + (pcy - vcy) ** 2) ** 0.5)
        if score > best_score:
            best_score = score
            best = p
    return best


def draw_annotations(frame_rgb, plate_results, vehicle, matched_plate=None):
    """Draw coloured boxes on the frame — green for primary vehicle + matched plate, orange for others."""
    img = frame_rgb.copy()
    if vehicle:
        x1, y1, x2, y2 = vehicle["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(img, f"{vehicle['color']} {vehicle['brand']}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    for p in plate_results:
        if p["bbox"] is not None:
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            is_matched = matched_plate and p["plate"] == matched_plate["plate"]
            colour = (0, 200, 0) if is_matched else (255, 140, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(img, p["plate"], (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    return img


def draw_tracked(frame_rgb, tracks, plate_results, track_store):
    """Draw bounding boxes for all tracked vehicles + plates on the frame."""
    img = frame_rgb.copy()
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid  = t["track_id"]
        info = track_store.get(tid, {})
        label = f"#{tid}"
        if info.get("color") and info.get("brand"):
            label = f"#{tid} {info['color']} {info['brand']}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    for p in plate_results:
        if p["bbox"] is not None:
            px1, py1, px2, py2 = [int(v) for v in p["bbox"]]
            cv2.rectangle(img, (px1, py1), (px2, py2), (255, 140, 0), 2)
            cv2.putText(img, p["plate"], (px1, max(py1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 140, 0), 2)
    return img


def main():
    st.set_page_config(page_title="ALPR — Video Analysis", page_icon="🎥", layout="wide")
    st.title("🎥 Vehicle Identification — Video Mode")
    st.caption("Analyses a video file frame by frame · licence plate · colour · brand")

    alpr        = load_alpr()
    jordo_model, class_mapping = load_jordo23()
    detector    = load_yolo()

    st.sidebar.header("Video Input")

    # ── Source selection ──────────────────────────────────────────────
    source = st.sidebar.radio("Source:", ("Upload Video", "Select from folder"))

    video_path = None

    if source == "Upload Video":
        uploaded = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            # Save to a temp file so OpenCV can read it
            tmp_path = os.path.join("/tmp", uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            video_path = tmp_path
            st.sidebar.success(f"Loaded: {uploaded.name}")
    else:
        # Look for video files in the project folder
        video_dir = os.path.dirname(os.path.abspath(__file__))
        video_files = [f for f in os.listdir(video_dir)
                       if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        if video_files:
            sel = st.sidebar.selectbox("Select video:", video_files)
            video_path = os.path.join(video_dir, sel)
            st.sidebar.success(f"Loaded: {sel}")
        else:
            st.sidebar.warning("No video files found in project folder")

    # ── Settings ──────────────────────────────────────────────────────
    interval = st.sidebar.slider(
        "Analyse every N seconds", min_value=1, max_value=10, value=2,
        help="Higher = faster scrubbing, fewer detections"
    )

    if video_path is None:
        st.info("👆 Upload a video or select one from the sidebar to begin.")
        return

    # ── Video info ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total_frames / fps
    cap.release()

    st.markdown(f"**Duration:** {duration/60:.1f} min &nbsp;|&nbsp; **FPS:** {fps:.0f} &nbsp;|&nbsp; **Frames:** {total_frames:,}")

    # ── Scrub to any timestamp ────────────────────────────────────────
    seek_sec = st.slider("Jump to timestamp (seconds)", 0, int(duration), 0)

    col_frame, col_results = st.columns(2)
    frame_placeholder  = col_frame.empty()
    result_placeholder = col_results.empty()

    # ── Show a preview of the seek position ──────────────────────────
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec * 1000)
    ret, preview_frame = cap.read()
    cap.release()
    if ret:
        frame_placeholder.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB),
                                 caption=f"Preview @ {seek_sec}s", use_column_width=True)

    # ── Controls ─────────────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns(2)
    run_single = btn_col1.button("🔍 Analyse This Frame", type="primary",  use_container_width=True)
    run_video  = btn_col2.button("▶️ Process Full Video",  type="secondary", use_container_width=True)

    # ── Analyse single frame ──────────────────────────────────────────
    if run_single and ret:
        with st.spinner("Analysing frame..."):
            t0 = time.time()
            frame_rgb     = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            plate_results = process_plates(frame_rgb, alpr)
            tracks        = track_vehicles(preview_frame, detector)
            # Pick the largest tracked vehicle as the primary one
            primary_track = max(tracks, key=lambda t: t["norm_area"]) if tracks else None
            vehicle       = None
            matched_plate = None
            if primary_track:
                color, brand, brand_conf = classify_crop(primary_track["crop"], jordo_model, class_mapping)
                vehicle       = {"bbox": primary_track["bbox"], "color": color, "brand": brand,
                                 "confidence": primary_track["confidence"]}
                matched_plate = match_plate_to_vehicle(plate_results, vehicle)
            annotated = draw_tracked(frame_rgb, tracks, plate_results, {
                t["track_id"]: {"color": vehicle["color"] if vehicle else "",
                                "brand": vehicle["brand"] if vehicle else ""}
                for t in tracks
            })
            elapsed = (time.time() - t0) * 1000

        frame_placeholder.image(annotated, caption=f"Result @ {seek_sec}s", use_column_width=True)

        with result_placeholder.container():
            st.markdown(f"**⏱ Processed in {elapsed:.0f} ms**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### 🎨 Colour")
                st.success(f"**{vehicle['color']}**") if vehicle else st.warning("No vehicle")
            with c2:
                st.markdown("### 🏷️ Brand")
                st.info(f"**{vehicle['brand']}**") if vehicle else st.warning("No vehicle")
            with c3:
                st.markdown("### 🔢 Licence Plate")
                if matched_plate:
                    st.success(f"**{matched_plate['plate']}**")
                    st.caption(f"Confidence: {matched_plate['confidence']:.1%}")
                else:
                    st.warning("No plate detected")

    # ── Process full video with ByteTrack ──────────────────────────────
    if run_video:
        st.markdown("---")
        st.subheader("📋 Vehicle Log")

        log_placeholder = st.empty()
        log_rows        = []
        progress        = st.progress(0)
        status          = st.empty()
        frame_display   = st.empty()
        live_status     = st.empty()

        cap          = cv2.VideoCapture(video_path)
        frame_idx    = 0
        plate_counter = 0
        last_plate   = []
        vehicle_count = 0
        timestamp_sec = 0.0
        track_classify_count = {}  # { track_id: frame count for classify throttle }

        # Downscale factor — resize wide videos to MAX_VIDEO_WIDTH for faster YOLO
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        scale = min(1.0, MAX_VIDEO_WIDTH / raw_w) if raw_w > MAX_VIDEO_WIDTH else 1.0

        # ── Per-track accumulators  { track_id: {"frames": [...], "last_seen": int} }
        track_store   = {}   # stores best color/brand so far per track_id for display
        track_accum   = {}   # { track_id: [(color, brand, plate), ...] }
        track_last    = {}   # { track_id: frame_idx when last seen }
        finalised_ids = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx    += 1
            timestamp_sec = frame_idx / fps

            if frame_idx % 30 == 0:   # update UI every 30 frames
                status.text(f"Scanning {timestamp_sec:.0f}s / {duration:.0f}s  —  {vehicle_count} vehicle(s) logged")
                progress.progress(min(frame_idx / total_frames, 1.0))

            # Downscale frame if video is wider than MAX_VIDEO_WIDTH
            if scale < 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            # ── ByteTrack: get all tracked vehicles in this frame ──
            tracks = track_vehicles(frame, detector)

            # ── Plate detection every N frames ──
            plate_counter += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if plate_counter % PLATE_EVERY_N == 0:
                last_plate = process_plates(frame_rgb, alpr)

            active_ids = set()

            for t in tracks:
                tid  = t["track_id"]
                crop = t["crop"]
                active_ids.add(tid)
                track_last[tid] = frame_idx

                if tid in finalised_ids:
                    continue   # already logged this vehicle, skip

                # Throttle classification — only run every CLASSIFY_EVERY_N frames per track
                track_classify_count[tid] = track_classify_count.get(tid, 0) + 1
                if track_classify_count[tid] % CLASSIFY_EVERY_N != 1 and tid in track_accum:
                    continue   # skip inference this frame, use cached best

                # Classify this crop — returns confidence score too
                color, brand, brand_conf = classify_crop(crop, jordo_model, class_mapping)

                # Match a plate to this vehicle bbox
                fake_vehicle = {"bbox": t["bbox"]}
                matched_plate = match_plate_to_vehicle(last_plate, fake_vehicle)
                plate_text = matched_plate["plate"] if matched_plate else "—"

                # ── Keep highest-confidence prediction per track ──
                # First frame seeds the best; every new frame replaces it only if
                # its brand confidence is strictly higher than what we have so far.
                if tid not in track_accum:
                    track_accum[tid] = {"color": color, "brand": brand,
                                        "brand_conf": brand_conf, "plate": plate_text,
                                        "frames_seen": 1}
                else:
                    track_accum[tid]["frames_seen"] += 1
                    if brand_conf > track_accum[tid]["brand_conf"]:
                        track_accum[tid]["color"]      = color
                        track_accum[tid]["brand"]      = brand
                        track_accum[tid]["brand_conf"] = brand_conf
                    # Always update plate if we have a real one
                    if plate_text != "—":
                        track_accum[tid]["plate"] = plate_text

                # Keep best label for live display
                track_store[tid] = {
                    "color": track_accum[tid]["color"],
                    "brand": track_accum[tid]["brand"],
                    "conf":  track_accum[tid]["brand_conf"],
                }

                # ── Once we have seen enough frames — lock and log ──
                if track_accum[tid]["frames_seen"] >= FRAMES_TO_LOCK:
                    best        = track_accum[tid]
                    final_color = best["color"]
                    final_brand = best["brand"]
                    final_plate = best["plate"]
                    brand_conf  = best["brand_conf"]
                    vehicle_count += 1
                    finalised_ids.add(tid)

                    log_rows.append({
                        "#":          vehicle_count,
                        "Time":       f"{timestamp_sec:.0f}s",
                        "Colour":     final_color,
                        "Brand":      final_brand,
                        "Brand Conf": f"{brand_conf:.0%}",
                        "Plate":      final_plate,
                    })
                    log_placeholder.dataframe(log_rows, use_container_width=True)
                    live_status.success(
                        f"✅ Vehicle {vehicle_count} locked  ·  "
                        f"🎨 {final_color}  🏷️ {final_brand} ({brand_conf:.0%})  🔢 {final_plate}"
                    )

            # ── Check for tracks that disappeared — finalise if not yet logged ──
            gone_ids = [tid for tid, last in track_last.items()
                        if frame_idx - last > TRACK_LOST_FRAMES and tid not in finalised_ids]
            for tid in gone_ids:
                best = track_accum.get(tid)
                if best and best["frames_seen"] > 0:
                    final_color = best["color"]
                    final_brand = best["brand"]
                    final_plate = best["plate"]
                    brand_conf  = best["brand_conf"]
                    vehicle_count += 1
                    finalised_ids.add(tid)
                    log_rows.append({
                        "#":          vehicle_count,
                        "Time":       f"{timestamp_sec:.0f}s",
                        "Colour":     final_color,
                        "Brand":      final_brand,
                        "Brand Conf": f"{brand_conf:.0%}",
                        "Plate":      final_plate,
                    })
                    log_placeholder.dataframe(log_rows, use_container_width=True)

            # Show annotated frame every 5th frame to keep UI responsive
            if frame_idx % 5 == 0:
                annotated = draw_tracked(frame_rgb, tracks, last_plate, track_store)
                frame_display.image(annotated, caption=f"@ {timestamp_sec:.0f}s", use_column_width=True)

        cap.release()
        progress.progress(1.0)
        log_placeholder.dataframe(log_rows, use_container_width=True)
        status.success(f"Done — {vehicle_count} vehicle(s) in {duration/60:.1f} min video")


if __name__ == "__main__":
    main()
