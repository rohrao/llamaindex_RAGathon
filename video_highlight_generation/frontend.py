import streamlit as st
import requests

st.title("Video Upload and Display App")

# File upload widget
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

def call_process_video_api(file_content):
    url = "http://localhost:8000/process_video"
    files = {"input_file": ("video.mp4", file_content, "video/mp4")}
    rsp = requests.post(url, files=files)
    return rsp


if uploaded_file:
    st.video(uploaded_file)

    if st.button("Process Video"):
        # Display a progress bar for the video upload
        progress_bar = st.progress(0)
        uploaded_percentage = 0

        # Create a stream buffer to upload the video in chunks
        with uploaded_file:
            for chunk in uploaded_file.__iter__():
                uploaded_percentage += len(chunk) / uploaded_file.size
                progress_bar.progress(uploaded_percentage)
            file_content = uploaded_file.getvalue()

        url = "http://localhost:8000/process_video"

        response = call_process_video_api(file_content)

        if response.status_code == 200:
            st.success("Video processed successfully!")
            st.video(response.content)
        else:
            st.error("Error processing video. Please try again.")