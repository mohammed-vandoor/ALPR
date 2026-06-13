---
title: ALPR Vehicle Identification
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🚗 ALPR — Automatic Licence Plate Recognition & Vehicle Identification

A real-time system that identifies vehicles from CCTV footage or uploaded images.  
For each vehicle it tells you: **the licence plate number**, **the colour**, and **the brand**.

Built for car wash centres — when a car drives in, the system reads the plate and records the colour and brand automatically.

---

## What Does This System Do?

When you upload a photo (or in the future, connect a live camera), the system will:

1. **Find the car** in the image — even if there are multiple cars, it picks the main one (the largest / closest)
2. **Read the licence plate** — using AI-powered OCR (text recognition)
3. **Detect the colour** — using a trained AI model (15 colours: black, white, grey, silver, brown, beige, blue, red, green, yellow, orange, gold, purple, pink, tan)
4. **Identify the brand** — using a model trained on 8,949 vehicle types (BMW, Toyota, Ford, etc.)
5. **Show everything on screen** — colour-coded boxes drawn on the image, results shown as cards below

---

## How to Use It (Step by Step)

### Running Locally on Your Computer

```bash
# 1. Open a terminal and go to the project folder
cd /path/to/ALPR

# 2. Activate the environment (this loads all the required tools)
source venv/bin/activate

# 3. Start the app
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

### Using the App

1. On the left sidebar — choose **Upload Image** or **Select from test_images**
2. Pick your image
3. Click one of the three buttons:
   - **🔍 Detect Plates** — only reads licence plates
   - **🚗 Identify Vehicle** — only detects colour and brand
   - **⚡ Run Full Analysis** — does everything at once (recommended)
4. Results appear as three cards: **Colour · Brand · Licence Plate**

---

## Project File Structure

```
ALPR/
│
├── app.py                        ← Main app (the Streamlit web interface)
├── fast_vehicle_classifier.py    ← Colour detection using trained AI model
│
├── models/                       ← AI model files (not stored in GitHub — too large)
│   ├── yolo11n.pt                ← Detects vehicles in images (YOLO)
│   ├── color_classifier.pth      ← Our trained colour model (EfficientNet-B0)
│   └── color_classes.json        ← The 15 colour names the model knows
│
├── training/                     ← Scripts used to train the colour model
│   ├── train_color_v2.py         ← Main training script (EfficientNet-B0)
│   └── train_color_classifier.py ← Earlier version (MobileNetV3, kept for reference)
│
├── archive/                      ← Old experiments (not used in production)
│   ├── vehicle_detection.py      ← CLIP-based detector (replaced)
│   └── vlm_vehicle_classifier.py ← Florence-2 VLM (too slow for real-time)
│
├── test_images/                  ← Sample images to test the app
├── requirements.txt              ← Python packages needed to run the app
├── packages.txt                  ← System packages needed (for Linux servers)
├── runtime.txt                   ← Pins Python version to 3.11
└── README.md                     ← This file
```

---

## The AI Models — How Each One Works

### 1. Vehicle Detector — YOLO11n (`models/yolo11n.pt`)
- **What it does:** Scans the image and draws a box around every car, bus, or truck it finds
- **How it picks the "main" car:** Scores each car by size + how close to the bottom of the frame it is (closer = larger in frame = the car you care about)
- **Speed:** ~15ms per image on CPU

### 2. Colour Detector — EfficientNet-B0 (`models/color_classifier.pth`)
- **What it does:** Takes a crop of the car body and classifies its colour
- **How it was trained:** We merged 3 datasets totalling ~21,000 car images and fine-tuned a pretrained EfficientNet-B0 model on them
- **Colours it knows:** beige, black, blue, brown, gold, green, grey, orange, pink, purple, red, silver, tan, white, yellow
- **Accuracy:** ~88–92% on the test set
- **Speed:** ~5ms per crop on CPU
- **Where the model lives:** Hosted on HuggingFace Hub at `NihalVandoor/alpr-color-classifier` — downloaded automatically on first run

### 3. Brand Detector — Jordo23 EfficientNet-B4
- **What it does:** Classifies the car brand (make/model) from the vehicle crop
- **Source:** Hosted on HuggingFace at `Jordo23/vehicle-classifier`
- **Classes:** 8,949 vehicle types
- **Output:** Returns just the brand name e.g. "BMW", "Toyota", "Range Rover"
- **Speed:** ~200–400ms on CPU (downloaded automatically on first run)

### 4. Licence Plate Reader — fast-alpr
- **What it does:** Two steps — first detects where the plate is (YOLO-based), then reads the text (OCR)
- **Plate matching:** When multiple cars are in the image, it matches the plate to the primary vehicle using bounding box overlap, then proximity as a fallback
- **Annotated image:** The matched plate gets a **green box**, other plates get an **orange box**

---

## Training the Colour Model (For Developers)

The colour model was trained on Google Colab using a free T4 GPU. Here's how to retrain it if you want to improve accuracy or add new colours.

### Datasets Used
| Dataset | Images | Notes |
|---|---|---|
| VCoR (Vehicle Color Recognition) | ~7,200 | Road CCTV cameras |
| seebicb vehicle-color-recognition | ~7,100 | Road cameras |
| DataCluster vehicle-color-detection | ~6,600 | Annotated bounding boxes |

### Steps to Retrain

**1. Download the datasets** from Kaggle and put them in `data/` folder

**2. Run the merge + training script:**
```bash
python training/train_color_v2.py
```
This will:
- Merge all three datasets into one unified folder (`data/merged/<colour>/`)
- Apply data augmentation (random crop, colour jitter, blur) to simulate CCTV conditions
- Fine-tune EfficientNet-B0 for 12 epochs
- Save the model to `models/color_classifier.pth`

**3. Upload the new model to HuggingFace:**
```bash
hf auth login
hf upload NihalVandoor/alpr-color-classifier models/color_classifier.pth color_classifier.pth --repo-type model
```

### Training Settings
| Setting | Value | Why |
|---|---|---|
| Base model | EfficientNet-B0 | Fast + accurate, good for mobile/edge |
| Image size | 224×224 | Standard for EfficientNet |
| Epochs | 12 | Enough to converge without overfitting |
| Batch size | 32 | Fits in GPU memory |
| Optimizer | AdamW | Better generalisation than Adam |
| LR schedule | CosineAnnealing | Smooth decay, avoids sharp drops |
| Label smoothing | 0.1 | Prevents overconfidence on ambiguous colours |
| Augmentation | ColorJitter + GaussianBlur | Simulates CCTV lighting variation |

---

## Deploying to Streamlit Cloud

1. Push your code to GitHub
2. Go to **https://share.streamlit.io** → New app → connect your repo
3. Set main file to `app.py`
4. The app will automatically:
   - Install system packages from `packages.txt`
   - Install Python packages from `requirements.txt`
   - Download AI models from HuggingFace on first startup

> **Note:** The first startup takes 2–3 minutes while models download. After that it's cached.

---

## Known Limitations

- **Colour accuracy under bad lighting:** The model was trained on daytime road images. Very dark CCTV footage may return incorrect colours.
- **Brand detection speed:** The Jordo23 brand model is ~200–400ms on CPU. For real-time video streams, a GPU is recommended.
- **Licence plate OCR:** Works best on clear, front-facing plates. Heavily angled or dirty plates may not read correctly.
- **Multiple cars:** The system always picks one "primary" vehicle. If you need results for all cars in the frame, the code would need modification.

---

## Future Improvements

- Connect to live RTSP camera stream (e.g. IP cameras at car wash entry)
- Add database logging — store plate + colour + brand + timestamp per vehicle visit
- Upload `color_classifier.pth` re-trained on night/CCTV-specific images for better low-light accuracy
- Add make/model confidence threshold — skip brand result if confidence is too low
