import os
import json
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

# Target class IDs from COCO dataset (2: car, 5: bus, 7: truck)
VEHICLE_CLASS_IDS = [2, 5, 7]

# -------------------------------------------------------------------
# COLOR DETECTION — Trained MobileNetV3-Small (88% accuracy on VCoR)
# Falls back to LAB k-means if model file not found
# -------------------------------------------------------------------
_COLOR_MODEL = None
_COLOR_CLASSES = None
_COLOR_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "color_classifier.pth")
_CLASSES_PATH = os.path.join(os.path.dirname(__file__), "color_classes.json")

if os.path.exists(_MODEL_PATH) and os.path.exists(_CLASSES_PATH):
    try:
        with open(_CLASSES_PATH) as f:
            _COLOR_CLASSES = json.load(f)
        _ckpt = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
        arch = _ckpt.get("arch", "mobilenet_v3_small")
        if arch == "efficientnet_b0":
            _m = models.efficientnet_b0(weights=None)
        else:
            _m = models.mobilenet_v3_small(weights=None)
        _m.classifier[-1] = nn.Linear(_m.classifier[-1].in_features, len(_COLOR_CLASSES))
        _m.load_state_dict(_ckpt["model_state"])
        _m.eval()
        _COLOR_MODEL = _m
        print(f"Loaded colour classifier ({len(_COLOR_CLASSES)} classes)")
    except Exception as e:
        print(f"Colour model load failed, using fallback: {e}")


def detect_color_hsv(image_crop):
    """
    Car colour detection.
    Uses trained MobileNetV3 if available (~5ms), otherwise LAB k-means fallback.
    """
    try:
        h, w = image_crop.shape[:2]
        roi = image_crop[int(h*0.05):int(h*0.70), int(w*0.10):int(w*0.90)]

        if _COLOR_MODEL is not None:
            pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            with torch.inference_mode():
                logits = _COLOR_MODEL(_COLOR_TFM(pil).unsqueeze(0))
            idx = logits.argmax(1).item()
            return _COLOR_CLASSES[idx].capitalize()

        # ── Fallback: LAB k-means ──────────────────────────────────────
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        V, S = hsv[:, :, 2], hsv[:, :, 1]
        mask = (V > 30) & ~((V > 220) & (S < 15))
        pixels = lab[mask] if mask.sum() > 50 else lab.reshape(-1, 3)
        if len(pixels) > 800:
            pixels = pixels[np.random.choice(len(pixels), 800, replace=False)]
        k = min(3, len(pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 6, 1.0)
        _, labels, centers = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        dom = centers[np.bincount(labels.flatten()).argmax()]
        L, a, b = dom[0]*100/255, dom[1]-128, dom[2]-128
        palette = [("Black",10,0,0),("White",95,0,2),("Silver",75,0,1),("Grey",55,-6,-1),
                   ("Red",40,55,35),("Blue",35,5,-45),("Green",35,-25,20),("Yellow",85,-5,75),
                   ("Orange",60,35,55),("Brown",42,14,8),("Beige",80,5,12),("Purple",30,20,-25)]
        dists = [((L-p[1])**2+(a-p[2])**2+(b-p[3])**2)**0.5 for p in palette]
        return palette[dists.index(min(dists))][0]

    except Exception:
        return "Unknown"


# -------------------------------------------------------------------
# BRAND DETECTION — EfficientNet-B0 fine-tuned on Stanford Cars
# ~20ms on CPU vs ~3000ms for CLIP large
# -------------------------------------------------------------------
class FastVehicleDetector:
    def __init__(self):
        """
        Production-grade vehicle detector.
        - YOLO11n for detection (same as CLIP pipeline)
        - HSV for color (~1ms, no model)
        - EfficientNet-B0 via HuggingFace pipeline for brand (~50-100ms CPU)
        """
        # Main detector
        self.base_detector = YOLO("yolo11n.pt")

        # ViT fine-tuned on car makes & models (300+ classes, ~84% top-1 accuracy)
        # Returns clean labels like "BMW 3 Series", "Range Rover Sport", "Audi Q7"
        self.brand_classifier = pipeline(
            "image-classification",
            model="dima806/car_models_image_detection",
        )

    def _to_pil(self, image_crop):
        """Convert BGR numpy array to RGB PIL image."""
        return Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))

    def _format_label(self, label):
        """
        Extract brand only from labels like 'BMW 3 Series' -> 'BMW',
        'Range Rover Sport' -> 'Range Rover', 'Mercedes-Benz C-Class' -> 'Mercedes-Benz'.
        """
        two_word_brands = [
            "Range Rover", "Land Rover", "Mercedes-Benz", "Alfa Romeo",
            "Aston Martin", "Rolls-Royce",
        ]
        for brand in two_word_brands:
            if label.startswith(brand):
                return brand
        return label.split()[0] if label else label

    def classify(self, image_crop):
        """
        Run fast color + brand classification on a vehicle crop.
        Returns: (color, brand, full_label, elapsed_ms)
        """
        t0 = time.time()

        # Color — pure HSV, no model
        color = detect_color_hsv(image_crop)

        # Brand — ViT car model classifier
        try:
            pil_img = self._to_pil(image_crop)
            results = self.brand_classifier(pil_img, top_k=3)
            brand = self._format_label(results[0]['label'])
            top3_str = ", ".join(
                f"{self._format_label(r['label'])} ({r['score']:.0%})"
                for r in results
            )
        except Exception as e:
            brand = "Unknown"
            top3_str = f"Error: {e}"

        elapsed_ms = (time.time() - t0) * 1000
        return color, brand, top3_str, elapsed_ms

    def detect_vehicles(self, frame):
        """
        Full pipeline: detect → select primary → classify.
        Returns list of vehicle dicts (same schema as VehicleDetector).
        """
        vehicles = []
        results = self.base_detector(frame, verbose=False)

        if not results or results[0].boxes is None:
            return vehicles

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        # Select most-central vehicle — best represents the subject car
        fh, fw = frame.shape[:2]
        best_score = -1
        primary = None

        for box, cls, conf in zip(boxes, class_ids, confidences):
            if int(cls) in VEHICLE_CLASS_IDS and conf > 0.40:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                # Distance of box centre from frame centre (normalised)
                cx = ((x1 + x2) / 2) / fw - 0.5
                cy = ((y1 + y2) / 2) / fh - 0.5
                dist = (cx ** 2 + cy ** 2) ** 0.5          # 0 = perfect centre
                # Score: large + central wins; centrality weighted more
                score = (area / (fw * fh)) - 1.5 * dist
                if score > best_score:
                    best_score = score
                    primary = {
                        'coords': (x1, y1, x2, y2),
                        'crop': frame[y1:y2, x1:x2],
                        'confidence': float(conf)
                    }

        if primary and primary['crop'].size > 0:
            color, brand, top3_str, elapsed_ms = self.classify(primary['crop'])
            x1, y1, x2, y2 = primary['coords']
            vehicles.append({
                'bbox': (x1, y1, x2, y2),
                'color': color,
                'make_model': brand,
                'top3_brands': top3_str,
                'confidence': primary['confidence'],
                'elapsed_ms': elapsed_ms,
                'track_id': None,
                'is_closest': True
            })

        return vehicles

    def draw_vehicles(self, frame, vehicles):
        """Draw detection overlay — same style as VehicleDetector."""
        img = frame.copy()
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            elapsed = vehicle.get('elapsed_ms', 0)

            hud_label = f"{vehicle['color']} {vehicle['make_model']} ({elapsed:.0f}ms)"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 4)
            text_width = len(hud_label) * 13
            cv2.rectangle(img, (x1, y1 - 35), (x1 + text_width, y1), (0, 200, 255), -1)
            cv2.putText(img, hud_label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return img
