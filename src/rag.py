from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms import LLM

from embedding_model import SentenceTransformerEmbedding
from prompt_template import prompt

class RAG:
    _query_engine: RetrieverQueryEngine

    def __init__(self, index_path, llm: LLM):
        index = self._load_index_from_disk(index_path)
        retriever = index.as_retriever(similarity_top_k=5)
        response_synthesizer = get_response_synthesizer(
            llm=llm,  
            response_mode="compact", 
            verbose=True,
            text_qa_template=prompt,
        )
        self._query_engine = RetrieverQueryEngine(
            retriever=retriever,  
            response_synthesizer=response_synthesizer,
        )

            
    def query(self, query: str) -> str:
        return self._query_engine.query("What are the benefits of house guests?")

    @staticmethod
    def _load_index_from_disk(index_path: str) -> VectorStoreIndex:

        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index_struct = storage_context.index_store.index_structs()[0]
        embed_model = SentenceTransformerEmbedding()

        index = VectorStoreIndex(
            storage_context=storage_context, 
            embed_model=embed_model, 
            index_struct=index_struct,
        )

        return index


    

    






