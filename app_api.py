from fastapi import FastAPI

from chromadb import Client, Settings
from .clip_embeddings import ClipEmbeddingsfunction

client = Client(Settings(is_persistent=True, persist_directory="./clip_chroma"))

ef = ClipEmbeddingsfunction()
app = FastAPI()


@app.get("/retrieval/{path_image}")
async def retrieval_image(path_image):
    coll = client.get_collection(name = "clip", embedding_function = ef)
    result = coll.query(
        query_texts = path_image,
        include = ["documents", "metadatas"],
        n_results = 2
        )
    docs = result['documents'][0]
    descs = result["metadatas"][0]
    list_of_docs = []
    for doc, desc in zip(docs, descs):
        list_of_docs.append((list(desc.values())[0]))
    return {"message": list_of_docs}