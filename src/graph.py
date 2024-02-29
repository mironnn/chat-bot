"""Graph implementation."""
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import SystemMessage

model = ChatOllama(
    model="llama2:70b-chat", num_ctx=4096, base_url="http://localhost:17434"
)

