__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import huggingface_hub
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.tools.retriever import create_retriever_tool
from langchain.chains import RetrievalQA

from langchain_community.chat_models.ollama import ChatOllama
from langchain.agents import AgentType, Tool, initialize_agent


model = ChatOllama(
    model="llama2:13b", num_ctx=4096, base_url="http://localhost:11434"
)

model_name = "BAAI/bge-base-en-v1.5"
# model_kwargs = {'device': 'cuda'}
encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity
local_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    # model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="Represent this sentence for searching relevant passages:",
)

urls = [
    "https://www.nextiva.com/support/articles/request-password-reset.html#:~:text=If%20you%20have%20forgotten%20or,email%20to%20reset%20your%20password.",
    "https://en.wikipedia.org/wiki/Nextiva",
    "https://www.nextiva.com/support/articles/device-management-in-nextos.html",
    "https://www.nextiva.com/support/articles/sms-mms-faqs.html",
    "https://www.nextiva.com/support/articles/nextiva-voice-features.html"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=local_embeddings,
)
retriever = vectorstore.as_retriever()
data = RetrievalQA.from_chain_type(
    llm=model, chain_type="stuff", retriever=vectorstore.as_retriever()
)

tools: list[Tool] = [
    Tool(
        name="State of data QA System",
        func=data.run,
        description="useful for when you need to answer questions about the nextiva. Input should be a fully formed question.",
    ),
]

agent = initialize_agent(
    tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# TEST
logger.info(agent.run("how to send SMS?",))

# def retrieve(state):
#     """
#     Uses tool to execute retrieval.

#     Args:
#         state (messages): The current state

#     Returns:
#         dict: The updated state with retrieved docs
#     """
#     print("---EXECUTE RETRIEVAL---")
#     messages = state["messages"]
#     # Based on the continue condition
#     # we know the last message involves a function call
#     last_message = messages[-1]
#     # We construct an ToolInvocation from the function_call
#     action = ToolInvocation(
#         tool=last_message.additional_kwargs["function_call"]["name"],
#         tool_input=json.loads(
#             last_message.additional_kwargs["function_call"]["arguments"]
#         ),
#     )
#     # We call the tool_executor and get back a response
#     response = tool_executor.invoke(action)
#     function_message = FunctionMessage(content=str(response), name=action.tool)

#     # We return a list, because this will get added to the existing list
#     return {"messages": [function_message]}