import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load the cascade
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Haar Cascade file not found or could not be loaded.")
except Exception as e:
    st.error(f"Error loading Haar Cascade file: {e}")
    st.stop()

# Title of the app
st.title("Face Detection using Viola-Jones Algorithm")

# Instructions
st.write(
    """
    ### Instructions:
    1. Upload an image using the file uploader below.
    2. Adjust the parameters (minNeighbors and scaleFactor) to fine-tune the face detection.
    3. Choose the color of the rectangles around the detected faces.
    4. Click the 'Detect Faces' button to process the image.
    5. Save the processed image with detected faces to your device.
    """
)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display the original image
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
    
    # Color picker for rectangle color
    rect_color = st.color_picker("Choose the color of the rectangles", "#FF0000")
    rect_color_bgr = tuple(int(rect_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    
    # Sliders for minNeighbors and scaleFactor
    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.2)
    
    if st.button("Detect Faces"):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        
        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), rect_color_bgr, 2)
        
        # Display the processed image
        st.image(image, channels="BGR", caption="Processed Image with Detected Faces", use_column_width=True)
        
        # Save the processed image
        if st.button("Save Processed Image"):
            cv2.imwrite("processed_image.jpg", image)
            st.success("Image saved as 'processed_image.jpg'")