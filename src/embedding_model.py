import numpy as np
from sentence_transformers import SentenceTransformer, models
from transformers import PreTrainedTokenizerFast, LlamaTokenizer
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager 

# This model is not necessarily the same one as the LLM answering the questions later on.
MODEL_PATH = "/root/huggingface/llama3_1b" 

class SentenceTransformerEmbedding(BaseEmbedding):
    model: SentenceTransformer
    model_name: str = "sentence-transformer"
    embed_batch_size: int = 10
    callback_manager: CallbackManager = None
    num_workers: int = 1

    def __init__(
        self,
        model_name: str = "sentence-transformer",  # Model name for validation
        embed_batch_size: int = 10,  # Example batch size
        callback_manager: CallbackManager = None,  # Default None, but can use a callback manager if needed
        num_workers: int = 1,  # Default is None
    ):
        
        # Explicitly define the tokenizer to be able to set the padding token.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # The word embedding model creates an embedding per word in the text.
        word_embedding_model = models.Transformer(
            model_name_or_path=MODEL_PATH,
        )
        word_embedding_model.tokenizer = tokenizer

        # The pooling model reduces the dimensions to one embedding per text.
        pooling_model = models.Pooling(
            word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(), # size of input embeddings
            pooling_mode='mean',  # Choose between 'mean', 'cls', 'max', etc.
        )

        # The final model uses the word_embedding_model and pooling_model sequentially.
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], 
            device='cuda',
            tokenizer_kwargs={"padding": True, "pad_token": tokenizer.pad_token}
        )

        super().__init__(
            model_name=model_name, 
            embed_batch_size=embed_batch_size, 
            callback_manager=callback_manager, 
            num_workers=num_workers, 
            model=model
        )

    def _get_text_embedding(self, text: str) -> np.ndarray:
        # Return the embedding for a single text
        return self.model.encode(text, convert_to_tensor=True).cpu().numpy()

    def _get_query_embedding(self, query: str) -> np.ndarray:
        # Return the embedding for a query (similar to _get_text_embedding)
        return self.model.encode(query, convert_to_tensor=True).cpu().numpy()

    def _aget_query_embedding(self, query: str) -> np.ndarray:
        # Async version (calls the sync version for now)
        return self._get_query_embedding(query)