from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores import AstraDBVectorStore
from llama_index.llms import Ollama
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import ServiceContext
import os


def main():
    model = "stablelm2:latest"
    embed_model = "BAAI/bge-base-en"
    embed_model_dim = 768
    collection_name = "astra_v_table"
    token = os.environ['token']
    api_endpoint = os.environ['api_endpoint']

    embed_model = HuggingFaceBgeEmbeddings(model_name=embed_model)
    llm = Ollama(model=model, temperature=0)
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    astra_db_store = AstraDBVectorStore(
        token=token,
        api_endpoint=api_endpoint,
        collection_name=collection_name,
        embedding_dimension=embed_model_dim,
    )

    storage_context = StorageContext.from_defaults(vector_store=astra_db_store)

    index = VectorStoreIndex.from_vector_store(service_context=service_context,
                                               vector_store=astra_db_store)
    query = index.as_query_engine()

    prompt = input("Enter your query: ")
    response = query.query(prompt)
    print(response)


if __name__ == "__main__":
    main()