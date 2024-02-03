import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from llm import create_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

class LLMClient:
    def __init__(self, model_name="mixtral_8x7b", model_type="NVIDIA"):
        self.llm = create_llm(model_name, model_type)

    def chat_with_prompt(self, system_prompt, prompt):
        langchain_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{input}")])
        chain = langchain_prompt | self.llm | StrOutputParser()
        response = chain.stream({"input": prompt})

        return response

    def multimodal_invoke(self, b64_string, steer=False, creativity=0, quality=9, complexity=0, verbosity=8):
        message = HumanMessage(content=[{"type": "text", "text": "Describe this image in detail:"},
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_string}"},}])
        if steer:
            return self.llm.invoke([message], labels={"creativity": creativity, "quality": quality, "complexity": complexity, "verbosity": verbosity})
        else:
            return self.llm.invoke([message])

# Streamlit configuration
st.set_page_config(layout="wide")
st.sidebar.title("NeVA-22B Live Video Stream")

# Video stream settings
video_source = st.sidebar.selectbox("Select video source", ["Webcam", "YouTube"])
youtube_url = st.sidebar.text_input("Enter YouTube video URL", "")

# NeVA-22B model initialization
neva_22b = LLMClient(model_name="neva_22b")

# Function to get frames from the video stream
def get_frame(video_source):
    if video_source == "Webcam":
        cap = cv2.VideoCapture(0)
    elif video_source == "YouTube":
        cap = cv2.VideoCapture(youtube_url)

    if not cap.isOpened():
        st.error("Error: Unable to open video source.")
        return None

    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    else:
        st.error("Error: Unable to read frame from video source.")
        return None

# Function to invoke NeVA-22B using LLMClient
def multimodal_invoke(b64_string):
    # Invoke NeVA-22B model using LLMClient
    response = neva_22b.multimodal_invoke(b64_string)

    # Update session state with the result
    st.session_state.image_query = response.content

# Main Streamlit app
frame_interval = 5  # seconds
video_frame = st.empty()

# Initialize frame_counter in session state
if "frame_counter" not in st.session_state:
    st.session_state.frame_counter = 0

while True:
    # Get frame from video stream
    frame = get_frame(video_source)

    if frame is not None:
        # Display video frame
        video_frame.image(frame, channels="RGB")

        # Send frame to NeVA-22B model every frame_interval seconds
        if st.session_state.frame_counter % frame_interval == 0:
            # Convert frame to Image
            pil_image = Image.fromarray(frame.astype('uint8'), 'RGB')

            # Save the Image to a BytesIO object
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=20)

            # Convert the Image to base64 string
            b64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Invoke NeVA-22B model using the modified function
            multimodal_invoke(b64_string)
            st.write(st.session_state.image_query)
        st.session_state.frame_counter += 1

