import os
import cv2
import streamlit as st
from fast_alpr import ALPR
from PIL import Image
import numpy as np

def process_image(image_array):
    """Process image and return license plate results."""
    # Initialize FastALPR
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-s-v1-global-model",
    )
    
    # Run detection
    results = alpr.predict(image_array)
    
    processed_results = []
    if results:
        for result in results:
            plate_text = result.ocr.text if result.ocr else "UNKNOWN"
            confidence = result.ocr.confidence if result.ocr else 0.0
            
            # Handle confidence if it's a list or float
            if isinstance(confidence, list):
                confidence = confidence[0] if confidence else 0.0
            
            processed_results.append({
                'plate': plate_text,
                'confidence': confidence,
                'bbox': result.bbox if hasattr(result, 'bbox') else None
            })
    
    return processed_results

def draw_results(image_array, results):
    """Draw bounding boxes and plate text on the image."""
    img = image_array.copy()
    
    for result in results:
        if result['bbox'] is not None:
            x1, y1, x2, y2 = result['bbox']
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(img, f"{result['plate']}", (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return img

def main():
    st.set_page_config(
        page_title="License Plate Detector",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 License Plate Detector")
    st.markdown("Upload an image or select from the test_images folder to detect license plates.")
    
    # Initialize session state for ALPR model
    if 'alpr_initialized' not in st.session_state:
        st.session_state.alpr_initialized = False
    
    # Sidebar for image selection
    st.sidebar.header("Image Selection")
    
    # Option to upload or select from folder
    option = st.sidebar.radio(
        "Choose image source:",
        ("Upload Image", "Select from test_images folder")
    )
    
    image_array = None
    image_source = ""
    
    if option == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            image_source = uploaded_file.name
            st.sidebar.success(f"Loaded: {image_source}")
    
    else:
        # Select from test_images folder
        test_images_dir = "test_images"
        
        if os.path.exists(test_images_dir):
            valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
            image_files = [f for f in os.listdir(test_images_dir) 
                          if f.lower().endswith(valid_extensions)]
            
            if image_files:
                selected_file = st.sidebar.selectbox(
                    "Select an image:",
                    image_files
                )
                
                if selected_file:
                    img_path = os.path.join(test_images_dir, selected_file)
                    image_array = cv2.imread(img_path)
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    image_source = selected_file
                    st.sidebar.success(f"Loaded: {selected_file}")
            else:
                st.sidebar.warning("No images found in test_images folder.")
        else:
            st.sidebar.warning("test_images folder not found.")
    
    # Main content area
    if image_array is not None:
        st.subheader(f"Processing: {image_source}")
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Original Image")
            st.image(image_array, use_column_width=True)
        
        # Process button
        if st.button("🔍 Detect License Plates", type="primary"):
            with st.spinner("Detecting license plates..."):
                results = process_image(image_array)
            
            # Display results
            with col2:
                st.write("### Detection Results")
                
                if results:
                    # Draw results on image
                    result_image = draw_results(image_array, results)
                    st.image(result_image, use_column_width=True)
                    
                    # Display plate information
                    st.write("### Detected Plates")
                    for idx, result in enumerate(results, start=1):
                        st.success(f"""
                        **Plate #{idx}**: {result['plate']}  
                        **Confidence**: {result['confidence']:.1%}
                        """)
                else:
                    st.warning("No license plates detected in this image.")
                    st.image(image_array, use_column_width=True)
    else:
        st.info("👆 Please upload an image or select one from the sidebar to begin.")

if __name__ == "__main__":
    main()
