from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips
from typing import List, Tuple, Any

from starlette.responses import FileResponse

app = FastAPI()

origins = ["http://localhost", "http://localhost:3000"]  # Add the frontend URL here
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_segments(input_file: Any, segments: List[Tuple[str, str]]) -> str:
    clips = []

    for start_time, end_time in segments:
        clip = VideoFileClip(input_file).subclip(start_time, end_time)
        clips.append(clip)

    final_clip = concatenate_videoclips(clips)

    output_file = "output.mp4"
    video = final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

    return output_file


@app.post("/process_video")
async def process_video(input_file: UploadFile = File(...)) -> FileResponse:
    try:
        input_file_path = f"uploads/{input_file.filename}"
        base_path = Path(__file__).parent
        file_path = Path.joinpath(base_path, input_file_path)
        with open(file_path, "wb") as f:
            f.write(input_file.file.read())

        time_ranges = [("00:01:00", "00:01:23"), ("00:02:00", "00:02:08"), ("00:02:34", "00:02:43")]
        video_file = extract_segments(input_file_path, time_ranges)
        return FileResponse(video_file, media_type='video/mp4')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
