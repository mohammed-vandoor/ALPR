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
from vehicle_detection import VehicleDetector
from fast_vehicle_classifier import FastVehicleDetector, detect_color_hsv
from vlm_vehicle_classifier import VLMVehicleClassifier

def process_license_plates(image_array, alpr):
    """Process image and return license plate results."""
    # Run detection
    results = alpr.predict(image_array)
    
    processed_results = []
    if results:
        for result in results:
            plate_text = result.ocr.text if result.ocr else "UNKNOWN"
            confidence = result.ocr.confidence if result.ocr else 0.0
            
            # Handle confidence if it's a list or float
            if confidence is None:
                confidence = 0.0
            elif isinstance(confidence, list):
                confidence = confidence[0] if confidence and isinstance(confidence[0], (int, float)) else 0.0
            elif not isinstance(confidence, (int, float)):
                confidence = 0.0
            
            processed_results.append({
                'plate': plate_text,
                'confidence': confidence,
                'bbox': result.bbox if hasattr(result, 'bbox') else None
            })
    
    return processed_results

@st.cache_resource
def load_jordo23():
    """Load Jordo23 EfficientNet-B4 model (cached across sessions)."""
    path = hf_hub_download(repo_id="Jordo23/vehicle-classifier", filename="vehicle_classifier.pth")
    checkpoint = torch.load(path, map_location="cpu")
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=8949)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["class_mapping"]

def run_jordo23(image_bgr, model, class_mapping):
    """Run Jordo23 EfficientNet-B4 on a BGR image crop and return result dict."""
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    detector = YOLO("yolo11n.pt")
    results = detector(image_bgr, verbose=False)
    if not results or results[0].boxes is None:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    VEHICLE_CLASS_IDS = [2, 5, 7]

    max_area, primary = -1, None
    for box, cls, conf in zip(boxes, class_ids, confidences):
        if int(cls) in VEHICLE_CLASS_IDS and conf > 0.40:
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                primary = {'coords': (x1, y1, x2, y2), 'crop': image_bgr[y1:y2, x1:x2], 'confidence': float(conf)}

    if primary is None or primary['crop'].size == 0:
        return None

    crop = primary['crop']
    color = detect_color_hsv(crop)

    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    top3_probs, top3_idx = torch.topk(probs, 3)
    elapsed_ms = (time.time() - t0) * 1000

    top_label = class_mapping[top3_idx[0][0].item()]
    top3_str = ", ".join(
        f"{class_mapping[top3_idx[0][i].item()]} ({top3_probs[0][i].item()*100:.1f}%)"
        for i in range(3)
    )
    x1, y1, x2, y2 = primary['coords']
    return {
        'bbox': (x1, y1, x2, y2),
        'color': color,
        'make_model': top_label,
        'top3_brands': top3_str,
        'confidence': primary['confidence'],
        'elapsed_ms': elapsed_ms,
        'is_closest': True
    }

def draw_results(image_array, plate_results, vehicle_results):
    """Draw bounding boxes for license plates and vehicles on the image."""
    img = image_array.copy()
    
    # Draw vehicle detections (green boxes)
    for vehicle in vehicle_results:
        x1, y1, x2, y2 = vehicle['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        display_text = f"{vehicle['color']} | {vehicle['make_model']}"
        cv2.putText(img, display_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw license plate detections (blue boxes)
    for result in plate_results:
        if result['bbox'] is not None:
            x1, y1, x2, y2 = result['bbox']
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
            cv2.putText(img, f"{result['plate']}", (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return img

def main():
    st.set_page_config(
        page_title="Vehicle & License Plate Detector",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Vehicle & License Plate Detector")
    st.markdown("Upload an image or select from the test_images folder to detect vehicles and license plates.")
    
    # Initialize session state for models
    if 'alpr_model' not in st.session_state:
        with st.spinner("Loading ALPR model..."):
            st.session_state.alpr_model = ALPR(
                detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="cct-s-v1-global-model",
            )
    
    if 'vehicle_detector' not in st.session_state:
        with st.spinner("Loading vehicle detection models (CLIP)..."):
            st.session_state.vehicle_detector = VehicleDetector()

    if 'fast_detector' not in st.session_state:
        with st.spinner("Loading fast detection model (EfficientNet)..."):
            st.session_state.fast_detector = FastVehicleDetector()

    if 'jordo23_model' not in st.session_state:
        with st.spinner("Loading Jordo23 EfficientNet-B4 (8949 classes)..."):
            st.session_state.jordo23_model, st.session_state.jordo23_classes = load_jordo23()

    if 'vlm_detector' not in st.session_state:
        with st.spinner("Loading Florence-2 VLM (~550MB, one-time download)..."):
            st.session_state.vlm_detector = VLMVehicleClassifier()
        st.session_state.vlm_available = (
            st.session_state.vlm_detector.vlm is not None
        )
    
    # Sidebar for image selection
    st.sidebar.header("Image Selection")
    
    # Option to upload or select from folder
    option = st.sidebar.radio(
        "Choose image source:",
        ("Upload Image", "Select from test_images folder")
    )
    
    image_array = None
    image_source = ""
    
    if option == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            image_source = uploaded_file.name
            st.sidebar.success(f"Loaded: {image_source}")
    
    else:
        # Select from test_images folder
        test_images_dir = "test_images"
        
        if os.path.exists(test_images_dir):
            valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
            image_files = [f for f in os.listdir(test_images_dir) 
                          if f.lower().endswith(valid_extensions)]
            
            if image_files:
                selected_file = st.sidebar.selectbox(
                    "Select an image:",
                    image_files
                )
                
                if selected_file:
                    img_path = os.path.join(test_images_dir, selected_file)
                    image_array = cv2.imread(img_path)
                    if image_array is None:
                        st.sidebar.error(f"Failed to read image: {selected_file}")
                    else:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                        image_source = selected_file
                        st.sidebar.success(f"Loaded: {selected_file}")
            else:
                st.sidebar.warning("No images found in test_images folder.")
        else:
            st.sidebar.warning("test_images folder not found.")
    
    # Main content area
    if image_array is not None:
        st.subheader(f"Processing: {image_source}")
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Original Image")
            st.image(image_array, use_column_width=True)
        
        # Process buttons
        col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)

        with col_btn1:
            detect_plates = st.button("🔍 License Plates", type="primary")
        with col_btn2:
            detect_vehicles = st.button("🚗 CLIP Model", type="secondary")
        with col_btn3:
            detect_vehicles_fast = st.button("⚡ Fast Model", type="secondary")
        with col_btn4:
            vlm_ok = st.session_state.get('vlm_available', False)
            detect_vlm = st.button("🧠 VLM Model", type="secondary", disabled=not vlm_ok,
                                   help="Florence-2 VLM — only available locally (requires ~2GB RAM)")
        with col_btn5:
            compare_all = st.button("📊 Compare All", type="secondary")

        if detect_plates or detect_vehicles or detect_vehicles_fast or detect_vlm or compare_all:
            plate_results = []
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            if detect_plates:
                with st.spinner("Detecting license plates..."):
                    plate_results = process_license_plates(image_array, st.session_state.alpr_model)
                with col2:
                    st.write("### Detection Results")
                    result_image = draw_results(image_array, plate_results, [])
                    st.image(result_image, use_column_width=True)
                    if plate_results:
                        st.write("### Detected License Plates")
                        for idx, result in enumerate(plate_results, start=1):
                            st.success(f"**Plate #{idx}**: {result['plate']} — Confidence: {result['confidence']:.1%}")
                    else:
                        st.warning("No license plates detected.")

            elif detect_vehicles:
                with st.spinner("Running CLIP model..."):
                    vehicle_results = st.session_state.vehicle_detector.detect_vehicles(image_bgr)
                with col2:
                    st.write("### Detection Results (CLIP)")
                    result_image = draw_results(image_array, [], vehicle_results)
                    st.image(result_image, use_column_width=True)
                    for idx, v in enumerate(vehicle_results, 1):
                        st.success(f"**Color**: {v['color']}")
                        st.info(f"**Brand**: {v['make_model']}")

            elif detect_vehicles_fast:
                with st.spinner("Running Fast model..."):
                    vehicle_results = st.session_state.fast_detector.detect_vehicles(image_bgr)
                with col2:
                    st.write("### Detection Results (Fast)")
                    result_image = st.session_state.fast_detector.draw_vehicles(image_bgr, vehicle_results)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.image(result_image, use_column_width=True)
                    for idx, v in enumerate(vehicle_results, 1):
                        st.success(f"**Color**: {v['color']}")
                        st.info(f"**Brand**: {v['make_model']}")

            elif detect_vlm:
                with st.spinner("Running Florence-2 VLM (may take 15-20s on CPU)..."):
                    vehicle_results = st.session_state.vlm_detector.detect_vehicles(image_bgr)
                with col2:
                    st.write("### Detection Results (Florence-2 VLM)")
                    result_image = st.session_state.vlm_detector.draw_vehicles(image_bgr, vehicle_results)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.image(result_image, use_column_width=True)
                    for idx, v in enumerate(vehicle_results, 1):
                        st.success(f"**Color**: {v['color']}")
                        st.info(f"**Brand**: {v['make_model']}")

            elif compare_all:
                st.write("---")
                st.write("## 📊 Model Comparison")

                r1, r2, r3 = None, None, None
                t_clip, t_fast, t_jordo = 0, 0, 0

                with st.spinner("Running all 4 models (VLM may take 15-20s)..."):
                    t0 = time.time()
                    clip_results = st.session_state.vehicle_detector.detect_vehicles(image_bgr)
                    t_clip = (time.time() - t0) * 1000
                    r1 = clip_results[0] if clip_results else None

                    t0 = time.time()
                    fast_results = st.session_state.fast_detector.detect_vehicles(image_bgr)
                    t_fast = (time.time() - t0) * 1000
                    r2 = fast_results[0] if fast_results else None

                    t0 = time.time()
                    r3 = run_jordo23(image_bgr, st.session_state.jordo23_model, st.session_state.jordo23_classes)
                    t_jordo = (time.time() - t0) * 1000

                    t0 = time.time()
                    vlm_results = st.session_state.vlm_detector.detect_vehicles(image_bgr)
                    t_vlm = (time.time() - t0) * 1000
                    r4 = vlm_results[0] if vlm_results else None

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.markdown("### 🚗 CLIP")
                    if r1:
                        st.success(f"**Color**: {r1['color']}")
                        st.info(f"**Brand**: {r1['make_model']}")
                        st.metric("Time", f"{t_clip:.0f} ms")
                    else:
                        st.warning("No vehicle detected")

                with c2:
                    st.markdown("### ⚡ Fast ViT")
                    if r2:
                        st.success(f"**Color**: {r2['color']}")
                        st.info(f"**Brand**: {r2['make_model']}")
                        st.metric("Time", f"{t_fast:.0f} ms")
                    else:
                        st.warning("No vehicle detected")

                with c3:
                    st.markdown("### 🏆 Jordo23")
                    if r3:
                        st.success(f"**Color**: {r3['color']}")
                        st.info(f"**Brand**: {r3['make_model']}")
                        st.metric("Time", f"{t_jordo:.0f} ms")
                    else:
                        st.warning("No vehicle detected")

                with c4:
                    st.markdown("### 🧠 Florence-2 VLM")
                    if r4:
                        st.success(f"**Color**: {r4['color']}")
                        st.info(f"**Brand**: {r4['make_model']}")
                        st.metric("Time", f"{t_vlm:.0f} ms")
                    else:
                        st.warning("No vehicle detected")
    else:
        st.info("👆 Please upload an image or select one from the sidebar to begin.")

if __name__ == "__main__":
    main()
