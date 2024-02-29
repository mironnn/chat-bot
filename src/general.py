"""main."""
import re
import sys
from pathlib import Path
from typing import Callable, Union

# from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.memory import FileChatMessageHistory
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langserve import add_routes  # type: ignore
from langserve.pydantic_v1 import BaseModel  # type: ignore
from loguru import logger  # type: ignore

# load_dotenv()

# logger.remove(0)
# logger.add(sys.stdout, level="DEBUG", serialize=False)

model = ChatOllama(
    model="llama2:70b-chat", num_ctx=4096, base_url="http://localhost:11434"
)


def _is_valid_identifier(value: str) -> bool:
    """Check if the session ID is in a valid format."""
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))


def create_session_factory(
    base_dir: Union[str, Path],
) -> Callable[[str], BaseChatMessageHistory]:
    base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir
    if not base_dir_.exists():
        base_dir_.mkdir(parents=True)

    def get_chat_history(session_id: str) -> FileChatMessageHistory:
        """Get a chat history from a session ID."""
        if not _is_valid_identifier(session_id):
            raise HTTPException(
                status_code=400,
                detail=f"Session ID `{session_id}` is not in a valid format. "
                "Session ID must only contain alphanumeric characters, "
                "hyphens, and underscores.",
            )
        file_path = base_dir_ / f"{session_id}.json"
        return FileChatMessageHistory(str(file_path))

    return get_chat_history


# app = FastAPI(
#     title="LangChain Server",
#     version="1.0",
#     description="Spin up a simple api server using Langchain's Runnable interfaces",
# )


prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an respectful and honest assistant. You have to answer the user's "
            "questions using only the context provided to you. If you don't know the answer, "
            "just say you don't know. Don't try to make up an answer.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{human_input}"),
    ]
)


chain: Runnable = prompt | model


class InputChat(BaseModel):
    """Input for the chat endpoint."""

    human_input: str


chain_with_history = RunnableWithMessageHistory(
    chain,
    create_session_factory(".chat_histories"),
    input_messages_key="human_input",
    history_messages_key="history",
).with_types(input_type=InputChat)


logger.info(
    chain_with_history.invoke(
        {"human_input": "Hello. how ar eyou?"},
        {"configurable": {"session_id": "77"}},
    )
)


# add_routes(
#     app,
#     chain_with_history,
# )

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=18000)
