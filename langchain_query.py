from dotenv import load_dotenv
load_dotenv()  # noqa

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os


EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
connection_args = {"uri": "http://milvus:19530"}

COLLECTION_NAME = 'milvus_docs_collection'

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
)

memory = ConversationBufferMemory(memory_key="history", input_key="question")

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
    base_url='http://localhost:11434',
    model=LLM_MODEL_NAME
    # n_ctx=2048,             # Context window size
    # temperature=0.7,        # Sampling temperature
    # top_p=0.95,             # Top-p sampling
    # # Include any other valid parameters
)


def ask_question(question):
    retrieved_docs = vector_store.similarity_search(question, k=4)
    context = '\n\n'.join([doc.page_content for doc in retrieved_docs])
    history = memory.load_memory_variables({}).get('history', '')
    prompt_text = prompt.format(context=context, history=history, question=question)
    response = llm(prompt_text)
    memory.save_context({'question': question}, {'response': response})
    return response


question = 'Explain IVF_FLAT in Milvus.'
response = ask_question(question)
print(response)
print('='*100, '\n')
question2 = 'List all index building parameters their short description and their default values.'
response2 = ask_question(question2)
print(response2)

"""
In Milvus, IVF_FLAT is a type of index that divides a vector space into list clusters. When the default list value (16,384) is used, it works similarly to FLAT. 

However, unlike FLAT, IVF_FLAT first compares the distances between the target vector and the centroids of all 16,384 clusters to return probe nearest clusters. Then it compares the distances between the target vector and the vectors in the selected clusters to get the nearest vectors.

IVF_FLAT is beneficial when the number of vectors exceeds the default list value (16,384) by a factor of two or more, as it starts to show performance advantages compared to FLAT.
==================================================================================================== 

Based on the provided context, here are the index building parameters with their short descriptions and default values:

1. drop_ratio_build:
   - Description: The proportion of small vector values excluded during indexing process
   - Range: [0, 1]
   - Default Value: Not specified in the text.

2. drop_ratio_search:
   - Description: The proportion of small vector values excluded during search process
   - Range: [0, 1]
   - Default Value: Not specified in the text.


Note that the default values for these parameters are not explicitly provided in the given text, but they can be found by referring to ${MILVUS_ROOT_PATH}/configs/milvus.yaml
"""