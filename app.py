import os
import cv2
import time
import torch
import timm
import streamlit as st
from fast_alpr import ALPR
from PIL import Image
import numpy as np
from torchvision import transforms
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from fast_vehicle_classifier import detect_color_hsv

VEHICLE_CLASS_IDS = [2, 5, 7]
YOLO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolo11n.pt")


@st.cache_resource
def load_alpr():
    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-s-v1-global-model",
    )


@st.cache_resource
def load_jordo23():
    path = hf_hub_download(repo_id="Jordo23/vehicle-classifier", filename="vehicle_classifier.pth")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=8949)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
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


def detect_primary_vehicle(image_bgr, detector):
    results = detector(image_bgr, verbose=False)
    if not results or results[0].boxes is None:
        return None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    fh, fw = image_bgr.shape[:2]
    best_score, primary = -1, None
    for box, cls, conf in zip(boxes, class_ids, confidences):
        if int(cls) in VEHICLE_CLASS_IDS and conf > 0.40:
            x1, y1, x2, y2 = map(int, box)
            norm_area = (x2 - x1) * (y2 - y1) / (fw * fh)
            norm_bottom = y2 / fh
            score = norm_area + 0.5 * norm_bottom
            if score > best_score:
                best_score = score
                primary = {"coords": (x1, y1, x2, y2), "crop": image_bgr[y1:y2, x1:x2], "confidence": float(conf)}
    return primary


def run_vehicle_detection(image_bgr, jordo_model, class_mapping, detector):
    primary = detect_primary_vehicle(image_bgr, detector)
    if primary is None or primary["crop"].size == 0:
        return None

    crop = primary["crop"]
    color = detect_color_hsv(crop)

    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0)

    with torch.no_grad():
        logits = jordo_model(tensor)
    top_idx = torch.softmax(logits, dim=1).argmax(1).item()
    brand = extract_brand(class_mapping[top_idx])

    x1, y1, x2, y2 = primary["coords"]
    return {"bbox": (x1, y1, x2, y2), "color": color, "brand": brand, "confidence": primary["confidence"]}


def process_plates(image_array, alpr):
    results = alpr.predict(image_array)
    out = []
    for r in (results or []):
        conf = r.ocr.confidence if r.ocr else 0.0
        if conf is None: conf = 0.0
        elif isinstance(conf, list): conf = conf[0] if conf else 0.0
        out.append({"plate": r.ocr.text if r.ocr else "UNKNOWN", "confidence": float(conf),
                    "bbox": r.bbox if hasattr(r, "bbox") else None})
    return out


def draw_annotations(image_rgb, plate_results, vehicle):
    img = image_rgb.copy()
    if vehicle:
        x1, y1, x2, y2 = vehicle["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(img, f"{vehicle['color']} {vehicle['brand']}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    for p in plate_results:
        if p["bbox"] is not None:
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 100, 0), 3)
            cv2.putText(img, p["plate"], (int(x1), max(int(y1) - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)
    return img


def main():
    st.set_page_config(page_title="ALPR — Vehicle Identifier", page_icon="🚗", layout="wide")
    st.title("🚗 Vehicle Identification System")
    st.caption("Detects licence plate · colour · brand from images or CCTV stream")

    alpr       = load_alpr()
    jordo_model, class_mapping = load_jordo23()
    detector   = load_yolo()

    st.sidebar.header("Image Input")
    option = st.sidebar.radio("Source:", ("Upload Image", "Select from test_images"))

    image_array = None
    image_source = ""

    if option == "Upload Image":
        uploaded = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            image_array = np.array(Image.open(uploaded))
            image_source = uploaded.name
            st.sidebar.success(f"Loaded: {image_source}")
    else:
        test_dir = "test_images"
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            if files:
                sel = st.sidebar.selectbox("Select image:", files)
                img = cv2.imread(os.path.join(test_dir, sel))
                if img is not None:
                    image_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image_source = sel
                    st.sidebar.success(f"Loaded: {sel}")

    if image_array is not None:
        st.subheader(f"📷 {image_source}")
        col_img, col_out = st.columns(2)
        with col_img:
            st.image(image_array, caption="Input", use_column_width=True)

        col1, col2, col3 = st.columns(3)
        run_plates  = col1.button("🔍 Detect Plates",  type="primary",    use_container_width=True)
        run_vehicle = col2.button("🚗 Identify Vehicle", type="secondary", use_container_width=True)
        run_both    = col3.button("⚡ Run Full Analysis", type="secondary", use_container_width=True)

        if run_plates or run_vehicle or run_both:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            plate_results, vehicle = [], None

            if run_plates or run_both:
                with st.spinner("Detecting licence plates..."):
                    plate_results = process_plates(image_array, alpr)

            if run_vehicle or run_both:
                with st.spinner("Identifying vehicle..."):
                    t0 = time.time()
                    vehicle = run_vehicle_detection(image_bgr, jordo_model, class_mapping, detector)
                    elapsed = (time.time() - t0) * 1000

            annotated = draw_annotations(image_array, plate_results, vehicle)
            with col_out:
                st.image(annotated, caption="Results", use_column_width=True)

            st.markdown("---")
            res_cols = st.columns(3)

            with res_cols[0]:
                st.markdown("### 🎨 Colour")
                if vehicle:
                    st.success(f"**{vehicle['color']}**")
                else:
                    st.warning("No vehicle detected")

            with res_cols[1]:
                st.markdown("### 🏷️ Brand")
                if vehicle:
                    st.info(f"**{vehicle['brand']}**")
                    st.caption(f"Confidence: {vehicle['confidence']:.0%}  |  {elapsed:.0f} ms")
                else:
                    st.warning("No vehicle detected")

            with res_cols[2]:
                st.markdown("### 🔢 Licence Plate")
                if plate_results:
                    for p in plate_results:
                        st.success(f"**{p['plate']}**")
                        st.caption(f"Confidence: {p['confidence']:.1%}")
                else:
                    st.warning("No plate detected")
    else:
        st.info("👆 Upload an image or select one from the sidebar to begin.")


if __name__ == "__main__":
    main()
