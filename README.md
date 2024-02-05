# [LlamaIndex RAG-A-THON](https://rag-a-thon.devpost.com/)

Repo for the code of Rag-A-thon!\
[StreamLens Pitch](https://docs.google.com/presentation/d/1BsIpg1FHrwX1pq6H2uhMJ1a3o3MubCxy0ukkbzqdYCQ/edit?usp=sharing), 
[Demo Video](https://drive.google.com/file/d/1on5CQ4ma4eI43TdyT7YGPW7PH9K1CFqC/view?usp=sharing)

## Inspiration
* Every year, a staggering amount of content is streamed across various platforms:
* NFL, soccer, cricket, and more, contribute to thousands of hours of sports content.
* Webinars and events add to the growing pool of educational and entertainment material.

Currently, the creation of video content and metadata is a manual, time-consuming process. We saw an opportunity to streamline this, making content more accessible and engaging.

## What it does
StreamLens revolutionizes how we interact with streaming and static video content. By leveraging advanced AI, it can understand, summarize, and generate highlights of video content in real-time. This includes:

- **Live Context Understanding**: Dynamically comprehends the content being streamed, providing immediate insights.
- **Key Moments Summarization** : Identifies and summarizes the most significant moments in any game or event.
- **Q&A Feature**: Allows users to ask questions and receive pinpointed answers, directing them to relevant moments in the content.
- **Automated Video Highlights Generation**: Creates highlight reels automatically, saving countless hours of manual editing.


## How we built it
We utilized a combination of cutting-edge AI models and user-friendly front-end technologies:

- **Models** : Integrated ASR (BentoML Cloud) for audio recognition, NeVA (Cloud API) for video analysis, and Mixtral (Cloud API) for a comprehensive multimedia content understanding.
- **Frontend**: Developed using Streamlit, ensuring an intuitive and responsive user experience.


## Challenges we ran into
- **API Deployment**: Mastering the deployment of BentoML cloud API for seamless model integration.
- **Unified Content Analysis**: Merging audio transcription and vision/scene understanding into a cohesive analysis tool.
- **Multimedia Processing**: Addressing technical challenges related to streaming, callbacks, and efficient processing of audio and video files.

## Accomplishments that we're proud of
- **Architecture**: Designed an innovative application architecture that seamlessly integrates multiple AI technologies.
- **Potential**: Demonstrated the transformative potential of applying AI to streaming content, enhancing user engagement and automating content curation.

## What we learned
- **AI Application**: The application of AI to streaming data presents unique challenges but offers significant rewards.
- **On-device Inference**: The advancements in on-device inference are opening new possibilities for real-time content analysis.

## What's next for Multimodal AI
- Enhancing the Q&A feature using semantic search to guide users directly to key moments, improving the interactive experience.
- Make the automated highlight/reel generation part of the UI, which are separate apps now.
- Refining AI models for greater accuracy and a broader range of content analysis capabilities.

## Instructions to run this project locally.

### To run the Stremlens live stream system

Make sure you have a api endpoint and token created for Astra DB and set it up as environment variables api_endpoint and token

Make sure you have an api key `NVIDIA_API_KEY` created for NVIDIA AI foundation and set it up as an environment variable

Run the `ayncllm.py` file by installing the required `requirements.txt` using streamlit run `ayncllm.py`

This should start the streamlit app which would intuitively understand the video live streamed and answer questions asked on it

### To run the Streamlens video highlights generator

Change your working directory to video_highlight_generation and install the requirements. Run the frontend using streamlit run ayncllm.py and the backend using `uvicorn backend:app --host 0.0.0.0 --port 8000` and you would be able to run the highlight generator system that generates highlights from the uploaded video

