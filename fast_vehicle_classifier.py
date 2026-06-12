import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

_COLOR_MODEL = None
_COLOR_CLASSES = None
_COLOR_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "models", "color_classifier.pth")
_CLASSES_PATH = os.path.join(os.path.dirname(__file__), "models", "color_classes.json")

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
    Car colour detection using trained EfficientNet-B0 (~5ms).
    Falls back to LAB k-means palette if model file not present.
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

        # Fallback: LAB k-means palette
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
