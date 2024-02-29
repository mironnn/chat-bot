"""main."""
from typing import List, Any

from langchain import hub
from langchain.agents import AgentExecutor, BaseSingleActionAgent, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.tools import StructuredTool, tool
from langchain.tools.render import render_text_description
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.tools import BaseTool
from langserve.pydantic_v1 import BaseModel, Field  # type: ignore
from loguru import logger  # type: ignore
from langchain_core.messages import SystemMessage


model_for_customer = ChatOllama(model="mistral:7b-instruct-v0.2-fp16", num_ctx=4096)


class AccountNumberModel(BaseModel):
    """Model for account number."""
    account_number: str = Field(description="customer account number")


@tool("check-account-status", args_schema=AccountNumberModel, return_direct=False)
def check_account_status(account_number: str) -> str:
    """Lookup the account status for a user: `active` or `not active` will be returned."""
    logger.info(f"checking account status for account_number: {account_number}")
    if account_number in ["100", "200"]:
        return "active"
    return "not active"


@tool("check-account-credit", args_schema=AccountNumberModel, return_direct=False)
def check_account_credit(account_number: str) -> str:
    """Lookup the account credit for a customer: Return exactly Amount of money or `no information`."""
    logger.info(f"checking account status for account_number: {account_number}")
    if account_number == "100":
        return "$1000"
    elif account_number == "200":
        return "$2000"
    return "no information"


tools: List[BaseTool | StructuredTool] = load_tools(
    ["llm-math"], llm=model_for_customer
)
tools.extend([check_account_status, check_account_credit])  # type: ignore

# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

model_with_stop = model_for_customer.bind(stop=["\nObservation"])
agent: BaseSingleActionAgent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | model_with_stop
    | ReActJsonSingleInputOutputParser()
)


def error_handler(error) -> str:
    """Handle parsing errors."""
    logger.error(f"Error: {error}")
    return "Sorry, I couldn't understand that."


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=error_handler,
)

# customer_prompt = """Your job is to help a user to check their account status and credit.
# You only have certain tools you can use. These tools require specific input.
# If you don't know the required input, then ask the user for it.
# If you are unable to help the user, you can """


# def get_customer_messages(messages) -> Any:
#     return [SystemMessage(content=customer_prompt)] + messages


# account_chain = get_customer_messages | agent_executor

# agent_executor.invoke({"input": "I'm user with account number 100. Can you please advice my account credit?"})
agent_executor.invoke({"input": "Hello, how are you?"})