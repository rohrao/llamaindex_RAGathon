# llamaindex_RAGathon

Repo for the code of Rag-A-thon!

[Demo Video](https://drive.google.com/file/d/1on5CQ4ma4eI43TdyT7YGPW7PH9K1CFqC/view?usp=sharing)

## Vector Store
Astra Db
test

This is the Readme file for running this project locally.

## To run the Stremlens live stream system

Make sure you have a api endpoint and token created for Astra DB and set it up as environment variables api_endpoint and token

Make sure you have an api key NVIDIA_API_KEY created for NVIDIA AI foundation and set it up as an environment variable

Run the ayncllm.py file by installing the required requirements.txt using streamlit run ayncllm.py

This should start the streamlit app which would intuitively understand the video live streamed and answer questions asked on it

## To run the Streamlens video highlights generator

Change your working directory to video_highlight_generation and install the requirements. Run the frontend using streamlit run ayncllm.py and the backend using uvicorn backend:app --host 0.0.0.0 --port 8000 and you would be able to run the highlight generator system that generates highlights from the uploaded video

