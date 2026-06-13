
import os
import json
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_COLOR_MODEL = None
_COLOR_CLASSES = None
_COLOR_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "models", "color_classifier.pth")
_CLASSES_PATH = os.path.join(os.path.dirname(__file__), "models", "color_classes.json")

# Fallback class list in case json file is missing
_FALLBACK_CLASSES = ["beige", "black", "blue", "brown", "gold", "green", "grey",
                     "orange", "pink", "purple", "red", "silver", "tan", "white", "yellow"]

os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)

if not os.path.exists(_CLASSES_PATH):
    with open(_CLASSES_PATH, "w") as _f:
        json.dump(_FALLBACK_CLASSES, _f)
    print("Created color_classes.json from fallback")

if not os.path.exists(_MODEL_PATH):
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        _dl = hf_hub_download(repo_id="NihalVandoor/alpr-color-classifier",
                              filename="color_classifier.pth")
        shutil.copy(_dl, _MODEL_PATH)
        print("Downloaded colour model from HuggingFace Hub")
    except Exception as e:
        print(f"Could not download colour model: {e}")

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
        _m.to(_DEVICE).eval()
        _COLOR_MODEL = _m
        print(f"Loaded colour classifier ({len(_COLOR_CLASSES)} classes)")
    except Exception as e:
        print(f"Colour model load failed, using fallback: {e}")


def detect_color_hsv(image_crop):
    """
    Car colour detection using trained EfficientNet-B0 (~5ms).
    Returns 'Unknown' if model is not loaded.
    """
    color, _ = detect_color_with_conf(image_crop)
    return color


def detect_color_with_conf(image_crop):
    """
    Car colour detection. Returns (color_name, confidence_float).
    """
    try:
        h, w = image_crop.shape[:2]
        roi = image_crop[int(h*0.05):int(h*0.70), int(w*0.10):int(w*0.90)]

        if _COLOR_MODEL is None:
            return "Unknown", 0.0

        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            logits = _COLOR_MODEL(_COLOR_TFM(pil).unsqueeze(0).to(_DEVICE))
        probs = torch.softmax(logits, dim=1)[0]
        idx   = probs.argmax().item()
        return _COLOR_CLASSES[idx].capitalize(), probs[idx].item()

    except Exception:
        return "Unknown", 0.0
