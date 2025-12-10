# LangChain wrapper

from typing import Any, List, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

class LangChainAdapter:
    """
    Adapter for the LLM Provider (Ollama) implementing the LLMPort.
    
    This adapter is specifically configured for the ilsp/Llama-Krikri-8B-Instruct model.
    It encapsulates the configuration complexity of the ChatOllama client, ensuring
    that the application layer receives a configured, ready-to-use model instance.
    """

    def __init__(
        self, 
        model_name: str = "ilsp/llama-krikri-8b-instruct", 
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        context_window: int = 8192,
        request_timeout: float = 120.0
    ):
        """
        Initialize the LLM adapter.
        
        Args:
            model_name: The tag of the model in Ollama. 
            base_url: The network URL of the Ollama service. In Docker, this 
                      will typically be 'http://ollama:11434'.
            temperature: Controls the randomness of the output. 0.1 is selected
                         for high factual grounding in RAG scenarios.
            context_window: The 'num_ctx' parameter. Sets the size of the 
                            prompt processing window.
            request_timeout: Timeout for generation requests, increased to handle
                             long chain-of-thought generations typical in tutoring.
        """
        self.model_name = model_name
        self.base_url = base_url
        
        # Instantiate the ChatOllama client.
        # We explicitly pass the 'num_ctx' parameter to override the default 2048 limit.
        # We also define the 'stop' tokens to strictly adhere to the Llama 3 format,
        # although Ollama's Modelfile usually handles this automatically.
        self._llm: BaseChatModel = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_ctx=context_window,
            request_timeout=request_timeout,
            # Llama 3 specific stop tokens to prevent generation overrun
            stop=["<|eot_id|>", "<|end_header_id|>"] 
        )

    def generate_response(self, messages: List) -> str:
        """
        Synchronous method to generate a response from a list of messages.
        
        Args:
            messages: A list of LangChain Message objects (System, Human, AI).
            
        Returns:
            The string content of the model's response.
        """
        response = self._llm.invoke(messages)
        return response.content

    async def agenerate_response(self, messages: List) -> str:
        """
        Asynchronous method to generate a response.
        
        This is crucial for the Chainlit UI integration, allowing the UI to 
        remain responsive while the model is generating tokens.
        
        Args:
            messages: A list of LangChain Message objects.
            
        Returns:
            The string content of the model's response.
        """
        # The ainvoke method returns an AIMessage object
        response = await self._llm.ainvoke(messages)
        return response.content

    def get_llm_instance(self) -> BaseChatModel:
        """
        Returns the underlying LangChain runnable instance.
        
        This method allows the application layer to use the model in higher-order
        constructs like RetrievalQA chains or Agents without needing to 
        re-initialize the client.
        """
        return self._llm