import cv2
import time
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

# Target class IDs from COCO dataset (2: car, 5: bus, 7: truck)
VEHICLE_CLASS_IDS = [2, 5, 7]

# -------------------------------------------------------------------
# COLOR DETECTION — Fast HSV rule-based (no model, ~1ms)
# More reliable than CLIP for color since color is physics, not semantics
# -------------------------------------------------------------------
def detect_color_hsv(image_crop):
    """
    Fast HSV-based color detection using masked region analysis.
    Focuses on the center 60% of the crop to avoid background/windows.
    Returns a color name string.
    """
    try:
        h, w = image_crop.shape[:2]

        # Focus on center region — avoids windows, tyres, background
        cx1, cy1 = int(w * 0.2), int(h * 0.2)
        cx2, cy2 = int(w * 0.8), int(h * 0.8)
        roi = image_crop[cy1:cy2, cx1:cx2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Compute median HSV — more robust than mean against outliers
        h_med = np.median(hsv[:, :, 0])
        s_med = np.median(hsv[:, :, 1])
        v_med = np.median(hsv[:, :, 2])

        # Brightness-first classification
        if v_med < 45:
            return "Black"
        if v_med > 210 and s_med < 40:
            return "White"
        if s_med < 35:
            if v_med > 150:
                return "Silver"
            return "Grey"

        # Hue-based classification (OpenCV H range: 0–180)
        if (0 <= h_med <= 10) or (165 <= h_med <= 180):
            return "Red"
        if 11 <= h_med <= 25:
            return "Orange"
        if 26 <= h_med <= 34:
            return "Yellow"
        if 35 <= h_med <= 85:
            return "Green"
        if 86 <= h_med <= 125:
            return "Blue"
        if 126 <= h_med <= 145:
            if v_med < 80:
                return "Dark Blue"
            return "Blue"
        if 146 <= h_med <= 164:
            return "Purple"

        return "Unknown"
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
        dima806 labels are already clean e.g. 'BMW 3 Series', 'Range Rover Sport'.
        Just return as-is.
        """
        return label

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

        # Select largest area vehicle
        max_area = -1
        primary = None

        for box, cls, conf in zip(boxes, class_ids, confidences):
            if int(cls) in VEHICLE_CLASS_IDS and conf > 0.40:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
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
