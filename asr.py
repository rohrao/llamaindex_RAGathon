# Follow https://github.com/ml-explore/mlx-examples/tree/main/whisper to install mlx w/ whisper
# The following model weight and config.json should be in "model/" directory
# !wget https://huggingface.co/mlx-community/whisper-medium.en-mlx/resolve/main/config.json?download=true
# !wget https://huggingface.co/mlx-community/whisper-medium.en-mlx/resolve/main/weights.npz?download=true

import whisper
def asr(audio_file, srt_name, model_path):
    output = whisper.transcribe(audio_file, word_timestamps=True, path_or_hf_repo=model_path)
    with open(srt_name, 'w') as f:
        for i, segment in enumerate(output['segments']):
            text = segment["text"].replace("\n", " ")
            f.write(f"{i+1}\n") 
            f.write(f"{segment['start']} --> {segment['end']}\n")
            f.write(f"{text}\n\n")
