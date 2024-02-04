from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import whisper
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.ingestion import IngestionPipeline
from llama_index.extractors import TitleExtractor, SummaryExtractor
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import MetadataMode
# from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
# from llama_index.embeddings import *
from llama_index.embeddings import HuggingFaceEmbedding
from llm import LLMClient
from llama_index.llms import Ollama
from llama_index import ServiceContext
from llama_index.vector_stores import AstraDBVectorStore
from llama_index import Document
from llama_index.text_splitter import TokenTextSplitter
from llama_index import set_global_service_context
from llama_index.llms import LangChainLLM
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
import asyncio



token = os.environ['token']
api_endpoint = os.environ['api_endpoint']

def create_pipeline_astra_db(llm_type='nvidia',embed_model='local',collection_name='video_transcript'):
    print("Loading Pipeline")
    if embed_model=='local':
        print("embed_model local")
        embed_model = "BAAI/bge-base-en"
        embed_model_dim = 768
        embed_model = HuggingFaceEmbedding(model_name=embed_model)
    elif embed_model=='nvidia':
        print("embed_model nviida")
        embed_model_dim = 1024
        embed_model = HuggingFaceEmbedding(model_name=embed_model)
    else:
        print("embed_model else")
        embed_model = HuggingFaceEmbedding(model_name=embed_model)
    
    if llm_type=='nvidia':
        print('llm nvidia')
        nvai_llm = ChatNVIDIA(model='llama2_70b')
        llm = LangChainLLM(llm=nvai_llm)
    elif llm_type=='ollama':
        print('llm_ollama')
        llm = Ollama(model='stablelm2', temperature=0.1)
    else:
        print('llm else')
        llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)


    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
    set_global_service_context(service_context) 
    astra_db_store = AstraDBVectorStore(
        token=token,
        api_endpoint=api_endpoint,
        collection_name=collection_name,
        embedding_dimension=embed_model_dim,
    )

    transformations = [
        SentenceSplitter(chunk_size=1024, chunk_overlap=100),
        # TitleExtractor(llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8),
        # SummaryExtractor(llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8),
        embed_model,
    ]
    # text_splitter = TokenTextSplitter(chunk_size=512)
    return IngestionPipeline(transformations=transformations,vector_store=astra_db_store)

class AudioData(BaseModel):
    audio_file_path: str
pipeline = create_pipeline_astra_db(llm_type='nvidia',collection_name="video_test_collection")
app = FastAPI()
@app.get("/status")
def read_status():
    return {"status": "running"}    

@app.post("/transcribe")
async def transcribe(audio_data:AudioData):
    return {"transcription": whisper.transcribe(audio_data.audio_file_path)["text"]}


@app.post("/transcribe_v2")
async def transcribe(audio_data:AudioData):

    return {"transcription": whisper.transcribe(audio_data.audio_file_path,path_or_hf_repo="mlx-community/whisper-medium-mlx")["text"]}

class ingest_data(BaseModel):
    text: str
    metadata:dict

@app.post("/ingest")
def ingest(ingest_data:ingest_data):
    doc = [Document(text=ingest_data.text,metadata=ingest_data.metadata)]
    pipeline.run(documents=doc,num_workers=4)
    return {"Done"}