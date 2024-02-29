"""main."""
import re
import sys
from typing import List

from dotenv import load_dotenv
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

load_dotenv()

logger.remove(0)
logger.add(sys.stdout, level="DEBUG", serialize=False)

# model = ChatOllama(model="llama2:70b-chat-q4_1", num_ctx=4096)
model = ChatOllama(model="mistral:7b-instruct-v0.2-fp16", num_ctx=4096)


class AccountNumberModel(BaseModel):
    """Model for account number."""

    account_number: str = Field(description="customer account number")


class SendEmailModel(BaseModel):
    """Model for account email address and email content."""

    account_email: str = Field(description="customer email address")
    content: str = Field(description="content of the email body")


def validate_email_address(email) -> bool:
    """Perform some basic email address validation."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    # Check if the email matches the pattern
    if re.match(pattern, email):
        return True
    return False


@tool("check-account-status", args_schema=AccountNumberModel, return_direct=False)
def check_account_status(account_number: str) -> str:
    """Lookup the account status for a user: `active` or `not active` will be returned."""
    logger.info(f"checking account status for account_number: {account_number}")
    if account_number in ["123", "456"]:
        return "active"
    return "not active"


@tool("check-account-credit", args_schema=AccountNumberModel, return_direct=False)
def check_account_credit(account_number: str) -> str:
    """Lookup the account credit for a user: Amount of money will be retuned."""
    logger.info(f"checking account status for account_number: {account_number}")
    if account_number == "123":
        return "$1000"
    elif account_number == "456":
        return "$2000"
    return "unknown"


# @tool("send-email", args_schema=SendEmailModel, return_direct=False)
# def send_email(account_email: str, content: str) -> str:
#     """Send email to the provided email address. Result will be provided as return value."""
#     if validate_email_address(account_email):
#         return "email was sent successfully"
#     return "email address not valid"


tools: List[BaseTool | StructuredTool] = load_tools(["llm-math"], llm=model)
# tools.extend([check_account_status, check_account_credit, send_email])
tools.extend([check_account_status, check_account_credit])  # type: ignore

# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
# import pdb; pdb.set_trace()
# addition_to_template = "Try to specify if some actions were successfull."
# prompt.messages[0].prompt.template = prompt.messages[0].prompt.template + addition_to_template
# logger.info(f"Promt: {prompt}")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

model_with_stop = model.bind(stop=["\nObservation"])
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
    agent=agent, tools=tools, verbose=True,
    handle_parsing_errors=error_handler,
)

for i in [
    "hello, how are you?",
    # "could you please help me?",

    # # True
    # "I'm user with account number 123. Can you check my account status?",
    # "I'm user with account number 456. Does my account active?",
    # # False
    # "I'm user with account number 100500. Does my account work correctly?",
    # "I'm user with account number 77. Is my account deactivated?",
    # # 1000
    # "I'm user with account number 123. Can you please advice my account credit?",
    # # 2000
    # "I'm user with account number 456. Can you please advice my account credit?",
    # # Unknown
    # "I'm user with account number 888. Can you please advice my account credit?",
    # "I'm user with account number foo-bar-123. Can you please advice my account credit?",
    # # Strong
    # "I'm user with account number 123. Can you please advice my account status and credit?",

    # "Send please email to test@foo.com",
    # "I'm user with account number 123. Can you check my account credit? Could please result to my email with address foobar.com. Thanks",
]:
    agent_executor.invoke({"input": i})