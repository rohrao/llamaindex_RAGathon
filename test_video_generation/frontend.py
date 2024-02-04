# streamlit_app.py

import streamlit as st
import requests

st.title("Video Cropper App")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)

    if st.button("Process Video"):
        files = {"file": uploaded_file}
        response = requests.post("http://localhost:8000/process_video", files=files)

        if response.status_code == 200:
            st.success("Video processed successfully!")
            st.video(response.json()["output_file"])
        else:
            st.error("Error processing video. Please try again.")
