# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List, Tuple
import imageio
import numpy as np

app = FastAPI()

origins = ["http://localhost", "http://localhost:3000"]  # Add the frontend URL here
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_segments(input_file: str, segments: List[Tuple[str, str]]) -> str:
    clips = []

    for start_time, end_time in segments:
        start_time = int(float(start_time) * 30)  # assuming 30 fps
        end_time = int(float(end_time) * 30)  # assuming 30 fps

        video = imageio.get_reader(input_file, 'ffmpeg')
        frames = [video.get_data(i) for i in range(start_time, end_time)]
        clip = imageio.mimsave('<bytes>', frames, 'mp4', fps=30)
        clips.append(clip)

    output_file = "output.mp4"
    imageio.mimsave(output_file, clips, 'mp4', fps=30)

    return output_file

@app.post("/process_video")
async def process_video(input_file: UploadFile = File(...)) -> dict:
    try:
        input_file_path = f"uploads/{input_file.filename}"
        with open(input_file_path, "wb") as f:
            f.write(input_file.file.read())

        time_ranges = [("60", "203"), ("480", "627"), ("1929", "2589")]  # in seconds
        output_file = extract_segments(input_file_path, time_ranges)

        return {"status": "success", "message": "Video processed successfully", "output_file": output_file}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))