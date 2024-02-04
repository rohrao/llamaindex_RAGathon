import threading

import pydub
from aiortc.contrib.media import MediaRecorder
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from streamlit_webrtc import webrtc_streamer
import streamlit as st
from llm import LLMClient
import time
from PIL import Image
from io import BytesIO
import base64
import cv2

# Streamlit configuration
st.set_page_config(layout="wide")
st.title("Livestream Copilot")
if "query" not in st.session_state:
    st.session_state.image_query=""

neva_22b = LLMClient(model_name="neva_22b")
mixtral = LLMClient(model_name="mixtral_8x7b")

buffer = {"current_img": None, "data_stream": None, "current_buffer": [], "summarization_stream": None, "summary": "", "last_api_call_time": 0}
buffer_lock = threading.Lock()

# Function to add text to the top right corner of the image
def add_text_to_image(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    # Get the size of the text to be added
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the position for the top right corner
    position = (image.shape[1] - text_size[0] - 10, 30)

    # Add text to the image
    cv2.putText(image, text, position, font, font_scale, color, thickness)

def process_frame_neva(frame):
    # Convert the av.VideoFrame to a NumPy array
    img = frame.to_ndarray(format="bgr24")

    pil_image = Image.fromarray(img[...,::-1].astype('uint8'), 'RGB')
    
    # Save the Image to a BytesIO object
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=20)

    # Convert the Image to base64 string    
    b64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Invoke NeVA-22B model using the modified function
    image_description_stream = neva_22b.streaming_multimodal_invoke("Describe what is happening in this image in a single sentence.", b64_string)
    
    # Update the buffer data stream
    with buffer_lock:
        buffer["data_stream"] = image_description_stream
        # Invoke summarization model on modified function
        summarization_stream = mixtral.chat_with_prompt(system_prompt="Your task is to summarize the content of a video stream. You will be given the summarization so far, and the new content to be incorporated. Provide only a single paragraph summary as a response. Do not reply with anything else.", prompt=" ".join(buffer["current_buffer"]))
        buffer["summarization_stream"] = summarization_stream

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with buffer_lock:
        buffer["current_img"] = frame.to_ndarray(format="rgb24")

    current_time = time.time()
    last_api_call_time = buffer.get("last_api_call_time", 0)
    elapsed_time = current_time - last_api_call_time
    if elapsed_time > 5:
        process_frame_neva(frame)
        with buffer_lock:
            buffer["current_buffer"] = []
            buffer["last_api_call_time"] = current_time
    return frame

webrtc_ctx = webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback)

col1, col2 = st.columns(2)
with col1:
    container = st.empty()

with col2:
    buffercontainer = st.empty()

if "audio_buffer" not in st.session_state:
    st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
if "video_capturing" not in st.session_state:
    st.session_state["video_capturing"] = []

# video_frames = []
while webrtc_ctx.state.playing:
    with buffer_lock:
        buffercontainer.empty()
        # video_stream = cv2.VideoCapture(0)
        # ret, frame = video_stream.read()
        # # Add the frame to the list
        # if ret:
        #     # video_frames.append(frame)
        #     st.session_state["video_capturing"].append(frame)
        # frame = webrtc_ctx.read_video_frame()
        # if frame is not None:
        #     video_frames.append(frame)
        # get the new summary
        current_img = buffer["current_img"]
        if current_img is not None:
            st.session_state["video_capturing"].append(current_img)
        if buffer["data_stream"]:
            try:
                buffer["current_buffer"].append(next(buffer["data_stream"]).content)
                add_text_to_image(current_img, " ".join(buffer["current_buffer"]))
            except StopIteration:
                pass
        if buffer["summarization_stream"]:
            try:
                buffer["summary"] += next(buffer["summarization_stream"])
                buffercontainer.write(buffer["summary"])
            except StopIteration:
                pass
        if current_img is not None:
            container.image(current_img, channels="BGR")
    time.sleep(0.1)
audio_buffer = st.session_state["audio_buffer"]
video_frames = st.session_state['video_capturing']
if len(video_frames) > 0:
        print(len(video_frames))
        clip = ImageSequenceClip(video_frames, fps=30)
            # Save the video
        clip.write_videofile("recorded_video.mp4", codec="libx264", fps=30)
        st.info("Video saved as 'recorded_video.mp4'")
# if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
# st.info("Writing wav to disk")
# pydub.AudioSegment.converter = '/opt/homebrew/bin/ffmpeg'
# audio_file = audio_buffer.export("temp.wav", format="wav")
# audio_file.close()
# video_stream.release()
# Convert frames to video
# while not webrtc_ctx.state.playing:
#     # with buffer_lock:
#         if len(video_frames) > 0:
#             print(len(video_frames))
#             clip = ImageSequenceClip(video_frames, fps=30)
#             # Save the video
#             clip.write_videofile("recorded_video.mp4", codec="libx264", fps=30)
#             st.info("Video saved as 'recorded_video.mp4'")
# auido_file = audio_buffer.export("temp.mp3", format="mp3")
# audio_buffer.export("temp.ogg", format="ogg")
# audio_buffer.export("temp.raw", format="raw")
#
#
# import asyncio
# import io
# import aiofiles
#
#
# async def record_video(webrtc_ctx, output_file):
#     local_stream = webrtc_ctx.local_streams[0]
#     video_track = local_stream.get_video_tracks()[0]
#
#     media_recorder = MediaRecorder(video_track)
#     await media_recorder.start(60)  # Record for 60 seconds, adjust as needed
#
#     chunks = []
#     while media_recorder.state == "recording":
#         chunk = await media_recorder.get_chunk()
#         chunks.append(chunk)
#
#     await media_recorder.stop()
#
#     blob = io.BytesIO()
#     for chunk in chunks:
#         blob.write(chunk.to_array_buffer())
#     blob.seek(0)
#
#     async with aiofiles.open(output_file, 'wb') as f:
#         while True:
#             data = await blob.read(8192)
#             if not data:
#                 break
#             await f.write(data)
#
# asyncio.run(record_video(webrtc_ctx, 'recorded_video.mp4'))