# ALPR — Automatic Licence Plate Recognition & Vehicle Classification

Real-time vehicle identification system for CCTV streams. Detects **number plates**, **colour**, and **brand** from live video or uploaded images.

---

## System Overview

```
CCTV / Image Upload
       │
       ▼
┌─────────────────┐
│  YOLO11n         │  Vehicle detection (bounding boxes)
│  yolo11n.pt      │  Classes: car, bus, truck
└────────┬────────┘
         │ vehicle crop
    ┌────┴────────────────────────────────┐
    │                                     │
    ▼                                     ▼
┌──────────────────┐          ┌──────────────────────┐
│  Colour Model     │          │  Brand Model          │
│  EfficientNet-B0  │          │  ViT / EfficientNet   │
│  color_classifier │          │  dima806 (300+ makes) │
│  15 colour classes│          │  + Jordo23 (8949 cls) │
└──────────────────┘          └──────────────────────┘
         │                                │
         └──────────────┬─────────────────┘
                        ▼
               ┌─────────────────┐
               │  fast-alpr       │
               │  Number Plate    │
               │  OCR + Detection │
               └─────────────────┘
                        │
                        ▼
              Streamlit Dashboard
         Colour | Brand | Plate Number
```

---

## Project Structure

```
ALPR/
├── app.py                      # Streamlit UI — main entry point
├── fast_vehicle_classifier.py  # Fast colour + brand pipeline (~20ms)
├── vehicle_detection.py        # CLIP-based classifier
├── vlm_vehicle_classifier.py   # Florence-2 VLM (local only)
│
├── models/
│   ├── yolo11n.pt              # YOLO11 nano detector
│   ├── color_classifier.pth    # Trained EfficientNet-B0 colour model
│   └── color_classes.json      # 15 colour class labels
│
├── training/
│   ├── train_color_v2.py       # Training script (EfficientNet-B0)
│   └── train_color_classifier.py # Earlier MobileNetV3 script (reference)
│
├── requirements.txt
├── packages.txt                # Streamlit Cloud system deps
└── .python-version             # Python 3.11 pin for Streamlit Cloud
```

---

## Colour Detection

### Model
- **Architecture**: EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
- **Classes (15)**: beige, black, blue, brown, gold, green, grey, orange, pink, purple, red, silver, tan, white, yellow
- **Accuracy**: ~88–92% on held-out validation set
- **Inference**: ~5–8ms per crop on CPU

### Training Data (21,000 images combined)
| Dataset | Images | Source |
|---|---|---|
| VCoR (Vehicle Color Recognition) | 7,267 | Kaggle — road CCTV cameras |
| seebicb vehicle-color-recognition | 7,100 | Kaggle — road cameras |
| DataCluster vehicle-color-detection | 6,633 | Kaggle — annotated bounding boxes |

### Training Pipeline (`training/train_color_v2.py`)
1. **Merge** all 3 datasets into unified colour folders (`data/merged/<colour>/`)
2. **Augmentation** (CCTV-optimised):
   - Random crop, horizontal flip
   - ColorJitter (brightness ±40%, contrast ±40%, saturation ±30%)
   - GaussianBlur to simulate camera quality variation
3. **Fine-tune** EfficientNet-B0 with AdamW + CosineAnnealing LR
4. **Label smoothing** (0.1) to prevent overconfidence on ambiguous colours

### To retrain
```bash
# 1. Download datasets from Kaggle
kaggle datasets download landrykezebou/vcor-vehicle-color-recognition-dataset -p data/vcor
kaggle datasets download seebicb/vehicle-color-recognition -p data/seebicb
kaggle datasets download dataclusterlabs/vehicle-color-detection-dataset -p data/dataclusterlabs

# 2. Extract
unzip data/vcor/*.zip -d data/vcor
unzip data/seebicb/*.zip -d data/seebicb
unzip data/dataclusterlabs/*.zip -d data/dataclusterlabs

# 3. Train (use Google Colab T4 GPU for ~8 min, or CPU for ~60 min)
python training/train_color_v2.py

# Output: models/color_classifier.pth + models/color_classes.json
```

---

## Brand Detection

### Models used
| Model | Classes | Accuracy | Speed |
|---|---|---|---|
| `dima806/car_models_image_detection` (ViT) | 300+ | ~84% top-1 | ~50ms CPU |
| `Jordo23/car-brand-classification` (EfficientNet) | 8,949 | ~78% | ~80ms CPU |

Both extract brand only (e.g. "BMW 3 Series" → "BMW").

---

## Number Plate Detection

Uses `fast-alpr` library — combines YOLO-based plate detector with OCR for plate text extraction.

---

## Running Locally

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## Streamlit Cloud Deployment

- Python version pinned to 3.11 via `.python-version`
- System package `libgl1` required for OpenCV (see `packages.txt`)
- **Note**: `color_classifier.pth` and `yolo11n.pt` are not in git (large files).
  The app falls back to LAB k-means colour detection if the trained model is not present.
  To enable the trained model on Streamlit Cloud, upload to HuggingFace Hub and add auto-download.

---

## Future Improvements
- Upload `color_classifier.pth` to HuggingFace Hub for cloud deployment
- Add real-time video stream support (RTSP/webcam)
- Fine-tune brand model on European car dataset
- Add make/model confidence threshold filtering
