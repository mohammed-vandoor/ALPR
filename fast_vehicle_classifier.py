import cv2
import time
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

# Target class IDs from COCO dataset (2: car, 5: bus, 7: truck)
VEHICLE_CLASS_IDS = [2, 5, 7]

# -------------------------------------------------------------------
# COLOR DETECTION — LAB k-means with perceptual palette matching (~5ms)
# -------------------------------------------------------------------
# Reference palette in LAB color space (perceptually uniform)
# Each entry: (name, L, a, b) — real-world car paint samples
# LAB distances match human color perception far better than HSV
# -------------------------------------------------------------------
_CAR_PALETTE_LAB = [
    ("Black",        10,   0,   0),
    ("White",        95,   0,   2),
    ("Silver",       75,   0,   1),
    ("Grey",         55,  -6,  -1),
    ("Dark Grey",    35,  -5,   0),
    ("Taupe",        60,  -5,   0),
    ("Red",          40,  55,  35),
    ("Dark Red",     28,  38,  22),
    ("Orange",       60,  35,  55),
    ("Yellow",       85,  -5,  75),
    ("Blue",         35,   5, -45),
    ("Dark Blue",    20,   5, -30),
    ("Light Blue",   60,  -5, -30),
    ("Green",        35, -25,  20),
    ("Dark Green",   22, -18,  12),
    ("Brown",        42,  14,   8),
    ("Dark Brown",   25,  15,   1),
    ("Bronze",       45,  10,  20),
    ("Copper Brown", 38,  18,  10),
    ("Beige",        80,   5,  12),
    ("Tan",          65,   8,  14),
    ("Purple",       30,  20, -25),
    ("Gold",         70,   5,  40),
    ("Maroon",       25,  25,  10),
]

# Pre-convert palette to numpy array for fast distance calc
_PALETTE_NAMES = [p[0] for p in _CAR_PALETTE_LAB]
_PALETTE_LAB = np.array([[p[1], p[2], p[3]] for p in _CAR_PALETTE_LAB], dtype=np.float32)


def detect_color_hsv(image_crop):
    """
    Accurate car color detection using:
    1. Centre crop + pixel masking to isolate body panels
    2. K-means (k=4) to find dominant color cluster
    3. LAB nearest-neighbor match against real car paint palette
    """
    try:
        h, w = image_crop.shape[:2]

        # Use top 65% of height — removes road/ground below the car
        # Horizontal: trim 10% each side to remove roadside clutter
        roi = image_crop[int(h*0.05):int(h*0.70), int(w*0.10):int(w*0.90)]

        # Convert to both HSV (for masking) and LAB (for color naming)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

        V = hsv[:, :, 2]
        S = hsv[:, :, 1]

        # Mask: remove very dark pixels (windows/tyres) and blown-out reflections
        body_mask = (V > 30) & ~((V > 220) & (S < 15))
        pixels_lab = lab[body_mask]

        if len(pixels_lab) < 50:
            pixels_lab = lab.reshape(-1, 3)

        # Subsample for speed
        if len(pixels_lab) > 2000:
            idx = np.random.choice(len(pixels_lab), 2000, replace=False)
            pixels_lab = pixels_lab[idx]

        # K-means on LAB pixels
        k = min(4, len(pixels_lab))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels_lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        # Largest cluster = dominant body color
        counts = np.bincount(labels.flatten())
        dominant_lab = centers[np.argmax(counts)]

        # Scale OpenCV LAB to real LAB range: L*[0,100], a*[-128,127], b*[-128,127]
        L = dominant_lab[0] * 100.0 / 255.0
        a = dominant_lab[1] - 128.0
        b = dominant_lab[2] - 128.0

        # Nearest-neighbor match in perceptual LAB space (Euclidean ΔE)
        query = np.array([L, a, b], dtype=np.float32)
        dists = np.linalg.norm(_PALETTE_LAB - query, axis=1)
        return _PALETTE_NAMES[int(np.argmin(dists))]

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
