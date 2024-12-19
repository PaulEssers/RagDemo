import gradio
from rag import RAG
from custom_llm import CustomLLM

INDEX_PATH = "data/gentleman"
LLM_URL = "http://localhost:8000/predict/"

custom_llm = CustomLLM(LLM_URL)
rag = RAG(INDEX_PATH, custom_llm)

# Create Gradio interface
interface = gradio.Interface(
    fn=rag.query,               # The function to call
    inputs=gradio.Textbox(label="Enter your query"),  
    outputs=gradio.Textbox(label="Answer"),  
    live=False                          
)

# Launch the interface
interface.launch(server_port=7860)