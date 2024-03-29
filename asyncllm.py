import threading

import pydub
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from streamlit_webrtc import webrtc_streamer
import streamlit as st
from llm import LLMClient
import time
from PIL import Image
from io import BytesIO
import base64
import cv2
from async_ingest import ingest_pipeline_astra_db
import pytz
from datetime import datetime
import requests

# Streamlit configuration
st.set_page_config(layout="wide")
st.title("Stream Lens Copilot")
if "query" not in st.session_state:
    st.session_state.image_query=""

if "video_on" not in st.session_state:
    st.session_state.video_on = False

if "summary" not in st.session_state:
    st.session_state.summary = ""

neva_22b = LLMClient(model_name="neva_22b")
mixtral = LLMClient(model_name="steerlm_llama_70b")

buffer = {"current_img": None, 
          "data_stream": None,
            "current_buffer": [], 
            "summarization_stream": None, 
            "summary": "", "last_api_call_time": 0,'ingest_summary':""}
buffer_lock = threading.Lock()
url = "http://localhost:8000/ingest"
start_tz = time.time()
start_time = time.time()



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
    image_description_stream = neva_22b.streaming_multimodal_invoke(f"""Continue describing what is happening in this image in a single sentence.
                                                                     Always highlight if something new is happening in the scene. 
                                                                    Do not provide any more information if it is not needed. 
                                                                    The summary of events thus far is as follows: {input_summary}""", b64_string)
    
    # Update the buffer data stream
    with buffer_lock:
        buffer["data_stream"] = image_description_stream
        # Invoke summarization model on modified function
        summarization_stream = mixtral.chat_with_prompt(system_prompt="""Your task is to summarize the content of a video stream.
                                                         You will be given the summarization so far, and the new content to be incorporated.
                                                         Provide only a single paragraph summary as a response. Do not reply with anything else.""", prompt=" ".join(buffer["current_buffer"]))
        buffer["summarization_stream"] = summarization_stream

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with buffer_lock:
        buffer["current_img"] = frame.to_ndarray(format="rgb24")

    current_time = time.time()
    last_api_call_time = buffer.get("last_api_call_time", 0)
    elapsed_time = current_time - last_api_call_time
    if elapsed_time > 5 and current_time - start_time > 5:
        process_frame_neva(frame, buffer["summary"])
        with buffer_lock:
            buffer["current_buffer"] = []
            buffer["last_api_call_time"] = current_time
    return frame

with st.sidebar:
    st.image("logo.png")
    st.title("Stream Lens")
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_frame_callback=video_frame_callback)
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
    st.session_state.video_on = True
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
                buffer["ingest_summary"]+= next(buffer["summarization_stream"]) 
                buffercontainer.write(buffer["summary"])
                st.session_state.summary = buffer["summary"]
            except StopIteration:
                pass
        if current_img is not None:
            container.image(current_img, channels="RGB")
    
    if len(buffer["ingest_summary"])>100:
        currenttz = pytz.timezone("America/Los_Angeles") 
        currenttime = datetime.now(currenttz)
        end_imestamp = currenttime.strftime("%Y-%m-%d_%H:%M:%S")
        metadata = {'end_tz': end_imestamp,'start_tz':start_tz}
        # ingest_pipeline_astra_db(buffer["ingest_summary"], metadata=metadata, _async=False, collection_name='test_collection',run_async=True)
        payload = {"text":buffer["ingest_summary"],
           "metadata":metadata}

        response = requests.post(url, json=payload)
        buffer["ingest_summary"]=""
        start_tz = end_imestamp
    time.sleep(0.1)
audio_buffer = st.session_state["audio_buffer"]
video_frames = st.session_state['video_capturing']
if not webrtc_ctx.state.playing:
    st.session_state.video_on = False
# if len(video_frames) > 0:
#         clip = ImageSequenceClip(video_frames[1:-1], fps=20)
#         # Save the video
#         clip.write_videofile("recorded_video.mp4", codec="libx264", fps=20)
#         st.info("Video saved as 'recorded_video.mp4'")

# chat = st.checkbox("Chat with the stream")
highlights = st.button("Generate highlights video")
if not st.session_state.video_on:
    st.subheader("Chat with your AI Assistant, Stream Lens!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", f"You are a helpful AI assistant named StreamLens. You have the ability to answer questions about video streams. You will respond to questions about the live video that the user just watched. If something is out of context, you will refrain from replying and politely decline to respond to the user. The video summary until this point is: {st.session_state.summary}. Note that if this is empty or blank, then please inform the user that they need to watch the streaming video before they can use the chatbot interface."), ("user", "{input}")]
    )
    user_input = st.chat_input("Can you tell me what happened in the video?")
    llm = LLMClient(model_name="mixtral_8x7b").llm

    chain = prompt_template | llm | StrOutputParser()

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # retriever = vectorstore.as_retriever()
        # docs = retriever.get_relevant_documents(user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        context = ""
        # for doc in docs:
            # context += doc.page_content + "\n\n"
        st.write(buffer["summary"])
        augmented_user_input = f"Question from user: " + user_input + "\n"

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for response in chain.stream({"input": augmented_user_input}):
                full_response += response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
