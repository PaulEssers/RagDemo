"""
This defines a model that is reachable through a REST API
"""
from llama_index.core.llms import LLMMetadata, LLM
from llama_index.core.callbacks import CallbackManager
from typing import AsyncIterator, Iterator, Optional
from pydantic import Field

class Output:
    text: str
    def __init__(self, text: str):
        self.text = text

class CustomLLM(LLM):
    endpoint: str

    def __init__(self, endpoint: str, callback_manager: Optional[CallbackManager] = None, **kwargs):
        """
        Parameters:
        ----------
        
        enpoint: Some REST api endpoint where the prompt can be POSTed to, which returns a response.
        """
        if callback_manager is None:
            callback_manager = CallbackManager()
        # supply enpoint to the super().__init__, because by defining this field, pydantic expects it,
        # even though it is not part of the Base LLM.
        super().__init__(endpoint=endpoint, callback_manager=callback_manager) 

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=2048,
            num_output_tokens=512,
            model_name="Custom Model"
        )

    def complete(self, prompt: str, **kwargs) -> dict:
        import requests
        response = requests.post(
            self.endpoint,
            json={"text": prompt}
        )
        response_data = response.json()

        return Output(response_data["result"])

    def chat(self, messages: list, **kwargs) -> str:
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        return self.complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, **kwargs) -> Iterator[str]:
        import requests
        with requests.post(
            self.endpoint,
            json={"text": prompt},
            stream=True
        ) as response:
            for chunk in response.iter_lines():
                if chunk:
                    yield chunk.decode("utf-8")

    def stream_chat(self, messages: list, **kwargs) -> Iterator[str]:
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        return self.stream_complete(prompt, **kwargs)

    async def acomplete(self, prompt: str, **kwargs) -> str:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json={"text": prompt}) as response:
                response_data = await response.json()
                return response_data["result"]

    async def achat(self, messages: list, **kwargs) -> str:
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        return await self.acomplete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json={"text": prompt}) as response:
                async for line in response.content:
                    yield line.decode("utf-8")

    async def astream_chat(self, messages: list, **kwargs) -> AsyncIterator[str]:
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        async for chunk in self.astream_complete(prompt, **kwargs):
            yield chunk