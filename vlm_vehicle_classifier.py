import cv2
import time
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fast_vehicle_classifier import detect_color_hsv

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    VLM_AVAILABLE = True
except Exception:
    VLM_AVAILABLE = False

VEHICLE_CLASS_IDS = [2, 5, 7]


class VLMVehicleClassifier:
    def __init__(self):
        """
        Local Vision Language Model classifier using Florence-2-base.
        ~270MB model, no system dependencies, runs on CPU in ~3-5s.
        Uses YOLO11 for detection + Florence-2 for vehicle description.
        """
        print("Loading YOLO11 detector...")
        self.detector = YOLO("yolo11n.pt")
        self.processor = None
        self.vlm = None

        if not VLM_AVAILABLE:
            print("Florence-2 unavailable: transformers import failed.")
            return

        try:
            print("Loading Florence-2-base VLM (~270MB)...")
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base",
                trust_remote_code=True
            )
            self.vlm = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            self.vlm.eval()
            print("Florence-2 loaded.")
        except Exception as e:
            print(f"Florence-2 failed to load: {e}")
            self.processor = None
            self.vlm = None

    def _to_pil(self, image_crop):
        img = cv2.resize(image_crop, (224, 224), interpolation=cv2.INTER_AREA)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _select_primary_vehicle(self, frame):
        """Run YOLO and return the most-central vehicle crop."""
        results = self.detector(frame, verbose=False)
        if not results or results[0].boxes is None:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        fh, fw = frame.shape[:2]

        best_score, primary = -1, None
        for box, cls, conf in zip(boxes, class_ids, confidences):
            if int(cls) in VEHICLE_CLASS_IDS and conf > 0.40:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                cx = ((x1 + x2) / 2) / fw - 0.5
                cy = ((y1 + y2) / 2) / fh - 0.5
                dist = (cx ** 2 + cy ** 2) ** 0.5
                score = (area / (fw * fh)) - 1.5 * dist
                if score > best_score:
                    best_score = score
                    primary = {
                        'coords': (x1, y1, x2, y2),
                        'crop': frame[y1:y2, x1:x2],
                        'confidence': float(conf)
                    }
        return primary

    def _extract_brand_from_caption(self, caption):
        """
        Parse Florence-2 caption to extract brand.
        e.g. 'A black BMW SUV parked on a street' -> 'BMW'
        """
        known_brands = [
            "Range Rover", "Land Rover", "BMW", "Mercedes-Benz", "Mercedes",
            "Audi", "Volkswagen", "VW", "Porsche", "Toyota", "Honda", "Ford",
            "Chevrolet", "Nissan", "Hyundai", "Kia", "Volvo", "Jaguar",
            "Lexus", "Tesla", "Mazda", "Subaru", "Jeep", "Peugeot",
            "Renault", "Citroën", "Citroen", "Skoda", "Seat", "Opel",
            "Fiat", "Alfa Romeo", "Ferrari", "Lamborghini", "Bentley",
            "Rolls-Royce", "Mini", "Maserati"
        ]
        caption_lower = caption.lower()
        for brand in known_brands:
            if brand.lower() in caption_lower:
                return brand
        return "Unknown"

    def classify(self, image_crop):
        """
        Run Florence-2 VLM on vehicle crop.
        Returns (color, brand_model, raw_caption, elapsed_ms)
        """
        t0 = time.time()
        color = detect_color_hsv(image_crop)

        if self.vlm is None or self.processor is None:
            return color, "VLM unavailable", "Florence-2 could not be loaded on this server.", (time.time() - t0) * 1000

        pil_image = self._to_pil(image_crop)

        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")

        with torch.inference_mode():
            generated_ids = self.vlm.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20,
                num_beams=1,
                do_sample=False,
            )

        raw = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Extract just the first sentence — usually "A <color> <brand> <type>"
        caption = raw.split(".")[0].strip()
        brand = self._extract_brand_from_caption(caption)

        elapsed_ms = (time.time() - t0) * 1000
        return color, brand, caption, elapsed_ms

    def detect_vehicles(self, frame):
        """Full pipeline: detect → select primary → VLM classify."""
        vehicles = []
        primary = self._select_primary_vehicle(frame)

        if primary is None or primary['crop'].size == 0:
            return vehicles

        color, brand, caption, elapsed_ms = self.classify(primary['crop'])
        x1, y1, x2, y2 = primary['coords']

        vehicles.append({
            'bbox': (x1, y1, x2, y2),
            'color': color,
            'make_model': brand,
            'top3_brands': caption,
            'confidence': primary['confidence'],
            'elapsed_ms': elapsed_ms,
            'track_id': None,
            'is_closest': True
        })
        return vehicles

    def draw_vehicles(self, frame, vehicles):
        """Draw VLM detection overlay — purple box to distinguish."""
        img = frame.copy()
        for v in vehicles:
            x1, y1, x2, y2 = v['bbox']
            elapsed = v.get('elapsed_ms', 0)
            label = f"{v['color']} {v['make_model']} ({elapsed:.0f}ms)"
            cv2.rectangle(img, (x1, y1), (x2, y2), (180, 0, 255), 4)
            text_width = len(label) * 13
            cv2.rectangle(img, (x1, y1 - 35), (x1 + text_width, y1), (180, 0, 255), -1)
            cv2.putText(img, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return img
