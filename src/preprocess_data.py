
import os
import numpy as np
import requests

from sentence_transformers import SentenceTransformer, models
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager 

from src.embedding_model import SentenceTransformerEmbedding

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Step 1: Download the book from Project Gutenberg
# The Gentlemen's Book of Etiquette and Manual of Politeness by Cecil B. Hartley
url = "https://www.gutenberg.org/cache/epub/39293/pg39293.txt"
response = requests.get(url)

if response.status_code == 200:
    text = response.text
else:
    print("Failed to download the text.")
    text = ""

# Step 2: Chunk the text into smaller parts
chunks = text.split('\r\n\r\n')  
chunks = [chunk.replace("\r\n", " ") for chunk in chunks if len(chunk) > 50 ]
chunks = chunks[11:-51]

# chunk = chunks[100]
# test_embedding = embedding_model.encode(chunk, convert_to_tensor=True)

# Step 3: Create the index
from llama_index.core import VectorStoreIndex, Document
documents = [Document(text=chunk) for chunk in chunks]
embed_model = SentenceTransformerEmbedding()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

index.storage_context.persist(persist_dir='data/gentleman/')