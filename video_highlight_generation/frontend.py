import streamlit as st
import requests
from io import BytesIO

st.title("Video Upload and Display App")

# File upload widget
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    st.video(uploaded_file)

    if st.button("Process Video"):
        # Display a progress bar for the video upload
        progress_bar = st.progress(0)
        uploaded_percentage = 0

        # Create a stream buffer to upload the video in chunks
        # video_stream = BytesIO()

        with uploaded_file:
            for chunk in uploaded_file.__iter__():
                # video_stream.write(chunk)
                uploaded_percentage += len(chunk) / uploaded_file.size
                progress_bar.progress(uploaded_percentage)

        # bytes_data = uploaded_file.read()
        # uploaded_file.seek(0)
        # files = {"file": uploaded_file}
        with open(uploaded_file.name, "rb") as f:
            f.seek(0)
            file_bytes = f.read()
            response = requests.post("http://localhost:8000/process_video", files={'vid.mp4': file_bytes})
        # response = requests.post("http://localhost:8000/process_video", files={'vid.mp4': open(uploaded_file, 'r')})
        # Reset the stream position before sending the request
        # video_stream.seek(0)

        # Send the video to the backend for processing
        # files = {"file": ("video_file", video_stream, "video/mp4")}

        if response.status_code == 200:
            st.success("Video processed successfully!")
            st.video(response.content)
        else:
            st.error("Error processing video. Please try again.")