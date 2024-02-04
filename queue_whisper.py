import streamlit as st
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer
# from streamlit_webrtc import VideoTransformerBase, VideoTransformerContext

from pydub import AudioSegment
import queue, pydub, tempfile, whisper, os, time

audio_model = whisper.load_model("base")

def save_audio(audio_segment: AudioSegment, base_filename: str) -> None:
    filename = f"{base_filename}_{int(time.time())}.wav"
    audio_segment.export(filename, format="wav")
    return filename

def transcribe(audio_segment: AudioSegment):
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
    # return sound


def handle_queue_empty(sound_chunk):
    if len(sound_chunk) > 0:
        text = transcribe(sound_chunk)
        st.write(text)
        sound_chunk = pydub.AudioSegment.empty()

    return sound_chunk

def app_sst(timeout=3):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
    )

    sound_chunk = pydub.AudioSegment.empty()

    while True:
        if webrtc_ctx.audio_receiver:
            # st.write("Running. Say something!")

            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=timeout)
            except queue.Empty:
                st.write("No frame arrived.")
                sound_chunk = handle_queue_empty(sound_chunk)
                continue
            
            # done to flush out old sound chunks and avoid concatenation and whisper-delays
            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                # this causes continuous concatenation
                sound_chunk = add_frame_to_chunk(audio_frame, sound_chunk)

            if len(sound_chunk) > 0:
                text = transcribe(sound_chunk)
                st.write(text)
        else:
            st.write("Stopping.")
            if len(sound_chunk) > 0:
                text = transcribe(sound_chunk.raw_data)
                st.write(text)
            break

def main():
    st.title("Real-time Speech-to-Text")   
    app_sst()

if __name__ == "__main__":
    main()
