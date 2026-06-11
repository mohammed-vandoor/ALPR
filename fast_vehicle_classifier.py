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
def _hsv_to_name(h, s, v):
    """Map median HSV values to a color name."""
    if v < 45:
        return "Black"
    if v > 200 and s < 40:
        return "White"
    if s < 45:
        return "Silver" if v > 150 else "Grey"
    if (0 <= h <= 10) or (165 <= h <= 180):
        return "Red"
    if 11 <= h <= 25:
        return "Orange"
    if 26 <= h <= 34:
        return "Yellow"
    if 35 <= h <= 85:
        return "Green"
    if 86 <= h <= 130:
        return "Dark Blue" if v < 90 else "Blue"
    if 131 <= h <= 164:
        return "Purple"
    return "Unknown"


def detect_color_hsv(image_crop):
    """
    Dominant body-color detection using k-means clustering on HSV.
    - Crops to centre 70% to remove background
    - Masks out near-black (windows/tyres) and near-white (sky/reflections)
    - Runs k-means (k=3) on remaining pixels, picks the largest cluster
    """
    try:
        h, w = image_crop.shape[:2]

        # Centre crop — removes most background
        roi = image_crop[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Mask out windows/tyres (very dark) and sky/reflections (very bright + unsaturated)
        body_mask = (V > 35) & ~((V > 195) & (S < 30))
        pixels = hsv[body_mask]

        if len(pixels) < 50:
            # Fallback to simple median if too few pixels survive masking
            h_m, s_m, v_m = np.median(H), np.median(S), np.median(V)
            return _hsv_to_name(h_m, s_m, v_m)

        # K-means clustering — find dominant color cluster
        samples = pixels.astype(np.float32)
        if len(samples) > 2000:
            idx = np.random.choice(len(samples), 2000, replace=False)
            samples = samples[idx]

        k = min(3, len(samples))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        # Pick the largest cluster
        counts = np.bincount(labels.flatten())
        dominant = centers[np.argmax(counts)]
        h_d, s_d, v_d = float(dominant[0]), float(dominant[1]), float(dominant[2])

        return _hsv_to_name(h_d, s_d, v_d)

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
