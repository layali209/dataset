import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.title("Face Detection App using Viola-Jones Algorithm")

st.markdown("""
### 📌 Instructions:
1. Upload an image.
2. Adjust detection parameters if needed.
3. Choose the rectangle color.
4. Click 'Detect Faces'.
5. Download the processed image with detected faces.
""")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

st.sidebar.header("Detection Settings")

min_neighbors = st.sidebar.slider(
    "minNeighbors",
    min_value=1,
    max_value=10,
    value=5
)

scale_factor = st.sidebar.slider(
    "scaleFactor",
    min_value=1.01,
    max_value=1.5,
    value=1.1,
    step=0.01
)

rectangle_color = st.sidebar.color_picker("Choose Rectangle Color", "#00FF00")

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  

color_bgr = hex_to_bgr(rectangle_color)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if uploaded_file is not None:
   
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Detect Faces"):
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), color_bgr, 2)

        st.image(image_np, caption="Detected Faces", use_column_width=True)

      
        output_path = "detected_faces.jpg"
        cv2.imwrite(output_path, image_np)

  
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Image",
                data=file,
                file_name="detected_faces.jpg",
                mime="image/jpeg"
            )

        st.success(f" {len(faces)} face(s) detected!")