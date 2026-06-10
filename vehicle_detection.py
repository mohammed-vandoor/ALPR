import cv2
import time
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

# Target class IDs from COCO dataset (2: car, 5: bus, 7: truck)
VEHICLE_CLASS_IDS = [2, 5, 7]

# Color candidates for CLIP zero-shot classification
# Descriptive prompts improve accuracy for dark/light variants
COLOR_LABELS = [
    "a white colored car",
    "a jet black colored car",
    "a dark black colored car",
    "a silver metallic car",
    "a grey colored car",
    "a dark grey colored car",
    "a red colored car",
    "a dark blue colored car",
    "a blue colored car",
    "a green colored car",
    "a yellow colored car",
    "an orange colored car",
    "a brown colored car",
    "a beige or cream colored car",
]
COLOR_NAMES = [
    "White", "Black", "Black", "Silver", "Grey", "Dark Grey",
    "Red", "Dark Blue", "Blue", "Green", "Yellow", "Orange", "Brown", "Beige"
]

# CLIP brand labels — works from any angle, globally
BRAND_LABELS = [
    "Range Rover", "Land Rover", "BMW", "Mercedes-Benz", "Audi",
    "Volkswagen", "Porsche", "Toyota", "Honda", "Ford", "Chevrolet",
    "Nissan", "Hyundai", "Kia", "Volvo", "Jaguar", "Lexus", "Tesla",
    "Mazda", "Subaru", "Jeep", "Peugeot", "Renault", "Citroën",
    "Skoda", "SEAT", "Opel", "Fiat", "Alfa Romeo", "Ferrari",
    "Lamborghini", "Maserati", "Bentley", "Rolls-Royce", "Mini"
]
BRAND_PROMPTS = [f"a {b} car" for b in BRAND_LABELS]

# Combined label list for a single CLIP inference pass
ALL_LABELS = COLOR_LABELS + BRAND_PROMPTS

class VehicleDetector:
    def __init__(self):
        """Initialize vehicle detection models with YOLO11 + HuggingFace classifiers."""
        # Main detector + tracking engine
        self.base_detector = YOLO("yolo11n.pt")
        
        # CLIP zero-shot classifier — used for both color and brand
        # Works from any angle, globally, no retraining needed
        self.clip_classifier = pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32"
        )
    
    def _to_pil(self, image_crop):
        """Convert BGR numpy array to PIL RGB image for HuggingFace pipeline."""
        rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    
    def classify_vehicle(self, image_crop):
        """
        Single CLIP inference pass for both color and brand.
        Splits results by label group after one forward pass.
        """
        t0 = time.time()
        try:
            pil_image = self._to_pil(image_crop)
            results = self.clip_classifier(pil_image, candidate_labels=ALL_LABELS)

            # Partition results back into color vs brand buckets
            color_scores = {r['label']: r['score'] for r in results if r['label'] in COLOR_LABELS}
            brand_scores = {r['label']: r['score'] for r in results if r['label'] in BRAND_PROMPTS}

            # Top color
            top_color_label = max(color_scores, key=color_scores.get)
            color = COLOR_NAMES[COLOR_LABELS.index(top_color_label)]

            # Top-3 brands
            top3_brand_items = sorted(brand_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_brand = BRAND_LABELS[BRAND_PROMPTS.index(top3_brand_items[0][0])]
            top3_str = ", ".join(
                f"{BRAND_LABELS[BRAND_PROMPTS.index(label)]} ({score:.0%})"
                for label, score in top3_brand_items
            )

            elapsed_ms = (time.time() - t0) * 1000
            return color, top_brand, top3_str, elapsed_ms
        except Exception as e:
            return "Unknown", "Unknown", "Unknown", 0
    
    def detect_vehicles(self, frame, confidence_threshold=0.40, use_tracking=False):
        """
        Detect vehicles in a frame and return vehicle information.
        
        Args:
            frame: Input image (BGR format)
            confidence_threshold: Minimum confidence for detection
            use_tracking: Whether to use YOLO11 tracking for persistent IDs
            
        Returns:
            List of dictionaries containing vehicle info (bbox, color, make_model, confidence, track_id)
        """
        vehicles = []
        
        if use_tracking:
            # Enable native YOLO11 ByteTRACK persistence engine
            results = self.base_detector.track(frame, persist=True, verbose=False)
        else:
            results = self.base_detector(frame, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                # Check if tracking is available
                has_tracking = result.boxes.id is not None if use_tracking else False
                
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int) if has_tracking else [None] * len(boxes)
                
                # Find the primary vehicle: largest bounding box area
                # (avoids partial/edge vehicles being selected over the main subject)
                closest_car = None
                max_area = -1
                
                for box, class_id, conf, track_id in zip(boxes, class_ids, confidences, track_ids):
                    if int(class_id) in VEHICLE_CLASS_IDS and conf > confidence_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        area = (x2 - x1) * (y2 - y1)
                        
                        if area > max_area:
                            max_area = area
                            closest_car = {
                                'coords': (x1, y1, x2, y2),
                                'id': track_id,
                                'crop': frame[y1:y2, x1:x2],
                                'confidence': conf
                            }
                
                # Process all vehicles or just the closest one
                vehicles_to_process = [closest_car] if closest_car else []
                
                for vehicle_data in vehicles_to_process:
                    x1, y1, x2, y2 = vehicle_data['coords']
                    track_id = vehicle_data['id']
                    vehicle_crop = vehicle_data['crop']
                    confidence = vehicle_data['confidence']
                    
                    if vehicle_crop.size == 0:
                        continue
                    
                    # Single CLIP pass for both color and brand
                    detected_color, top_brand, top3_brands, elapsed_ms = self.classify_vehicle(vehicle_crop)
                    make_model_label = top_brand
                    
                    vehicles.append({
                        'bbox': (x1, y1, x2, y2),
                        'color': detected_color,
                        'make_model': make_model_label,
                        'top3_brands': top3_brands,
                        'confidence': confidence,
                        'elapsed_ms': elapsed_ms,
                        'track_id': track_id,
                        'is_closest': True
                    })
        
        return vehicles
    
    def draw_vehicles(self, frame, vehicles):
        """
        Draw vehicle detection results on the frame with improved UI overlays.
        
        Args:
            frame: Input image (BGR format)
            vehicles: List of vehicle detection results
            
        Returns:
            Frame with drawn bounding boxes and labels with background cards
        """
        img = frame.copy()
        
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            is_closest = vehicle.get('is_closest', False)
            track_id = vehicle.get('track_id')
            
            if is_closest:
                # Draw Primary Target box in Bright Green
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                
                # Tag text composition
                if track_id is not None:
                    hud_label = f"TARGET #{track_id}: {vehicle['color']} {vehicle['make_model']}"
                else:
                    hud_label = f"{vehicle['color']} {vehicle['make_model']}"
                
                # Draw dark background card behind text for readability in high-glare environments
                text_width = len(hud_label) * 13
                cv2.rectangle(img, (x1, y1 - 35), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(img, hud_label, (x1 + 5, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                # Draw background queue cars in thin yellow boxes
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
                
                # Simple label for non-closest vehicles
                display_text = f"{vehicle['color']}"
                cv2.putText(img, display_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return img
