import os
import cv2
from fast_alpr import ALPR

def process_license_plates_in_folder(folder_path):
    """Loops through a folder of images and prints the discovered plate strings."""
    if not os.path.exists(folder_path):
        print(f"📁 Creating folder path: '{folder_path}'. Please drop your car pictures there.")
        os.makedirs(folder_path)
        return

    # Look for common static image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"⚠️ No images found inside the '{folder_path}' directory. Add some vehicle photos.")
        return

    print(f"⏳ Initializing FastALPR Models on CPU...")
    # open-image-models handles plate detection | fast-plate-ocr handles character recognition
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-s-v1-global-model",
    )

    print(f"🚀 Processing {len(image_files)} images...\n")

    for file_name in image_files:
        img_path = os.path.join(folder_path, file_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"❌ Error decoding file: {file_name}")
            continue

        print(f"📊 IMAGE FILE: {file_name}")
        
        # Run the image arrays directly through the models
        results = alpr.predict(frame)

        if not results:
            print("  └── 📇 No license plate detected.")
        else:
            for idx, result in enumerate(results, start=1):
                plate_text = result.ocr.text if result.ocr else "UNKNOWN"
                confidence = result.ocr.confidence if result.ocr else 0.0
                
                # Handle confidence if it's a list or float
                if isinstance(confidence, list):
                    confidence = confidence[0] if confidence else 0.0
                
                # Print clean, direct output to the console terminal
                if len(results) > 1:
                    print(f"  └── 📇 Plate #{idx}: {plate_text} ({confidence:.1%} Match)")
                else:
                    print(f"  └── 📇 Plate Text: {plate_text} ({confidence:.1%} Match)")
                    
        print("-" * 50)

    print("🏁 Batch processing folder completed.")

if __name__ == "__main__":
    # Define your target folder name next to the script
    TARGET_DIR = "test_images"
    
    process_license_plates_in_folder(TARGET_DIR)