from dotenv import load_dotenv
load_dotenv()  # noqa

from fastapi import FastAPI, Request
from pydantic import BaseModel
import os

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pymilvus import connections

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')
MILVUS_URI = os.getenv('MILVUS_URI')

app = FastAPI()


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
connection_args = {"uri": MILVUS_URI}
COLLECTION_NAME = 'milvus_docs_collection'

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
)

template = """
Hey, answer the user's question based on the following context:

The context is this: {context}

And this is the message history: {history}

The user's question is this: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "history", "question"],
    template=template,
)

llm = OllamaLLM(
    base_url='http://ollama:11434',  # Use the Docker service name
    model=LLM_MODEL_NAME,
)

memories = {}


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    question = request.message

    # Use a separate memory for each user
    if user_id not in memories:
        memories[user_id] = ConversationBufferMemory(memory_key="history", input_key="question")
    memory = memories[user_id]

    retrieved_docs = vector_store.similarity_search(question, k=4)
    context = '\n\n'.join([doc.page_content for doc in retrieved_docs])
    history = memory.load_memory_variables({}).get('history', '')
    prompt_text = prompt.format(context=context, history=history, question=question)
    response = llm(prompt_text)
    memory.save_context({'question': question}, {'response': response})

    return {"response": response}
