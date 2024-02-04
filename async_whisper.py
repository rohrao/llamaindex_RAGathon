import streamlit as st
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer
# from streamlit_webrtc import VideoTransformerBase, VideoTransformerContext

import threading
from pydub import AudioSegment
import queue, pydub, whisper, os, time

audio_model = whisper.load_model("base")
buffer = {"sound_chunk": pydub.AudioSegment.empty(), 
            "data_stream": None, 
            "current_buffer": [], 
            "summarization_stream": None, 
            "summary": "", 
            "last_api_call_time": 0}
buffer_lock = threading.Lock()


def save_audio(audio_segment: AudioSegment, base_filename: str):
    filename = f"{base_filename}_{int(time.time())}.wav"
    audio_segment.export(filename, format="wav")
    return filename

def whisper_transcribe(audio_segment: AudioSegment):
    current_filename = save_audio(audio_segment, "debug_audio")
    answer = audio_model.transcribe(current_filename, fp16=False)
    print("whisper_output=",answer["text"])
    with buffer_lock:
        buffer["data_stream"] = answer["text"]


def add_frame_to_chunk(audio_frame, sound_chunk):
    sound = pydub.AudioSegment(
        data=audio_frame.to_ndarray().tobytes(),
        sample_width=audio_frame.format.bytes,
        frame_rate=audio_frame.sample_rate,
        channels=len(audio_frame.layout.channels),
    )
    sound_chunk += sound
    return sound_chunk


def audio_frame_callback(frame):
    current_time = time.time()
    last_api_call_time = buffer.get("last_api_call_time", 0)
    elapsed_time = current_time - last_api_call_time
    if elapsed_time > 10:
        sound_chunk = buffer['sound_chunk']
        print("sound_chunk at Whisper=",len(sound_chunk))
        # calling in async fashion
        whisper_transcribe(sound_chunk)
        with buffer_lock:
            print("periodic_10sec_routine_ran")
            # buffer["sound_chunk"] = pydub.AudioSegment.empty()
            buffer["current_buffer"] = []
            buffer["last_api_call_time"] = current_time
    return frame


def app_sst():
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        audio_frame_callback=audio_frame_callback
    )

    while webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        print("audio_frames=",len(audio_frames))
        for audio_frame in audio_frames:
            # sound_chunk = buffer["sound_chunk"]
            # waiting for processed output could lead to delay but need to append every audio frame
            sound_chunk = add_frame_to_chunk(audio_frame, buffer["sound_chunk"])
            print("sound_chunk=",len(sound_chunk))
            with buffer_lock:
                buffer["sound_chunk"] = sound_chunk
 
        with buffer_lock:
            if buffer["data_stream"]:
                try:
                    st.write(buffer["data_stream"])
                except StopIteration:
                    pass
        time.sleep(1)

def main():
    st.title("Real-time Speech-to-Text")
    app_sst()

if __name__ == "__main__":
    main()
