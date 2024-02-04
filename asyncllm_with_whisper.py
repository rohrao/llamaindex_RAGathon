import threading

import pydub
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import streamlit as st
from llm import LLMClient
import time
from PIL import Image
from io import BytesIO
import base64
import cv2

import whisper, os
from pydub import AudioSegment
import queue

# Streamlit configuration
st.set_page_config(layout="wide")
st.title("Livestream Copilot")
if "query" not in st.session_state:
    st.session_state.image_query=""

neva_22b = LLMClient(model_name="neva_22b")
mixtral = LLMClient(model_name="mixtral_8x7b")

buffer = {"current_img": None, "data_stream": None, "current_buffer": [], "summarization_stream": None, "summary": "", "last_api_call_time": 0}
audio_chunks_buffer = []
buffer_lock = threading.Lock()

# Function to add text to the top right corner of the image
def add_text_to_image(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    # Get the size of the text to be added
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the position for the top right corner
    position = (image.shape[1] - text_size[0] - 10, 30)

    # Add text to the image
    cv2.putText(image, text, position, font, font_scale, color, thickness)

def process_frame_neva(frame, input_summary=""):
    # Convert the av.VideoFrame to a NumPy array
    img = frame.to_ndarray(format="bgr24")

    pil_image = Image.fromarray(img[...,::-1].astype('uint8'), 'RGB')
    
    # Save the Image to a BytesIO object
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=20)

    # Convert the Image to base64 string    
    b64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Invoke NeVA-22B model using the modified function
    image_description_stream = neva_22b.streaming_multimodal_invoke(f"Continue describing what is happening in this image in a single sentence. Always highlight if something new is happening in the scene. Do not provide any more information if it is not needed. The summary of events thus far is as follows: {input_summary}", b64_string)
    
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
        process_frame_neva(frame, buffer["summary"])
        with buffer_lock:
            buffer["current_buffer"] = []
            buffer["last_api_call_time"] = current_time
    return frame

### whisper section from queue_whisper
audio_model = whisper.load_model("tiny")

def save_audio(audio_segment: AudioSegment, base_filename: str):
    cwd = os.getcwd()
    filename = f"{cwd}/audio/{base_filename}_{int(time.time())}.wav"
    audio_segment.export(filename, format="wav")
    return filename

def transcribe(audio_segment: AudioSegment):
    print("Entered Whisper transcription")
    current_filename = save_audio(audio_segment, "debug_audio")
    answer = audio_model.transcribe(current_filename, fp16=False)
    print(answer["text"])
    # st.write(answer["text"])
    return answer["text"]

def add_frame_to_chunk(audio_frame, sound_chunk):
    sound = pydub.AudioSegment(
        data=audio_frame.to_ndarray().tobytes(),
        sample_width=audio_frame.format.bytes,
        frame_rate=audio_frame.sample_rate,
        channels=len(audio_frame.layout.channels),
    )
    sound_chunk += sound
    return sound_chunk

def handle_queue_empty(sound_chunk):
    if len(sound_chunk) > 0:
        text = transcribe(sound_chunk)
        st.write(text)
        sound_chunk = pydub.AudioSegment.empty()

    return sound_chunk





webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDONLY,
    video_frame_callback=video_frame_callback,
    audio_receiver_size=1024,
    media_stream_constraints={"video": True, "audio": True},    
    )

col1, col2 = st.columns(2)
with col1:
    container = st.empty()

with col2:
    buffercontainer = st.empty()

# Initializing streamlit session states
if "audio_buffer" not in st.session_state:
    st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
if "video_capturing" not in st.session_state:
    st.session_state["video_capturing"] = []

# video_frames = []
while webrtc_ctx.state.playing:
    text_payload = " ".join(audio_chunks_buffer)
    st.write("Audio transcription = ",text_payload)
    sound_chunk = pydub.AudioSegment.empty()
    if webrtc_ctx.audio_receiver:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=3)
        except queue.Empty:
            st.write("No frame arrived.")
            sound_chunk = handle_queue_empty(sound_chunk)
            continue
        
        # done to flush out old sound chunks and avoid concatenation and whisper-delays
        
        print("audio_frames=",audio_frames)
        for audio_frame in audio_frames:
            # this causes continuous concatenation
            sound_chunk = add_frame_to_chunk(audio_frame, sound_chunk)

        print("sound_chunk=",sound_chunk)
        if len(sound_chunk) > 0:
            text = transcribe(sound_chunk)
            audio_chunks_buffer.append(text)
            # text = mlx_transcribe(sound_chunk)
            # st.write(text)
    else:
        st.write("Stopping.")
        if len(sound_chunk) > 0:
            text = transcribe(sound_chunk.raw_data)
            audio_chunks_buffer.append(text)
            # text = mlx_transcribe(sound_chunk.raw_data)
            # st.write(text)
        break    
    with buffer_lock:
        buffercontainer.empty()
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
            container.image(current_img, channels="RGB")
    time.sleep(0.1)
audio_buffer = st.session_state["audio_buffer"]
video_frames = st.session_state['video_capturing']
if len(video_frames) > 0:
        clip = ImageSequenceClip(video_frames, fps=30)
        # Save the video
        clip.write_videofile("recorded_video.mp4", codec="libx264", fps=30)
        st.info("Video saved as 'recorded_video.mp4'")
