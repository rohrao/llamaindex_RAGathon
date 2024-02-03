# streamlit_app.py

import streamlit as st
import requests
import cv2
import numpy as np
from io import BytesIO

st.title("Video Cropper App")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)

    if st.button("Process Video"):
        files = {"file": uploaded_file}
        response = requests.post("http://localhost:8000/process_video", files=files)

        if response.status_code == 200:
            st.success("Video processed successfully!")

            # Display the processed video using OpenCV
            output_file = response.json()["output_file"]
            processed_video = cv2.VideoCapture(output_file)
            
            while True:
                ret, frame = processed_video.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, channels="RGB", use_column_width=True)

            processed_video.release()

        else:
            st.error("Error processing video. Please try again.")
