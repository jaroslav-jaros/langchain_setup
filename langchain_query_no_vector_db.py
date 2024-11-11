from dotenv import load_dotenv
load_dotenv()  # noqa

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os

LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')

memory = ConversationBufferMemory(memory_key="history", input_key="question")

template = """
Hey, answer the user's question.

And this is the message history: {history}

The user's question is this: {question}
"""

prompt = PromptTemplate(
    input_variables=["history", "question"],
    template=template,
)

llm = OllamaLLM(
    base_url='http://localhost:11434',  # Ollama server URL
    model=LLM_MODEL_NAME  # 'llama3.2:latest'
)


def ask_question(question):
    history = memory.load_memory_variables({}).get('history', '')
    prompt_text = prompt.format(history=history, question=question)
    response = llm(prompt_text)
    memory.save_context({'question': question}, {'response': response})
    return response


question = 'Explain IVF_FLAT in Milvus.'
response = ask_question(question)
print(response)

question2 = 'List all index building parameters their short description and their default values.'
response2 = ask_question(question2)
print(response2)

"""
I don't have any specific information on "IVF_FLAT" in the context of Milvus. However, I can provide some general information about Milvus and explain what it might relate to.

Milvus is an open-source, cloud-native graph data platform developed by Ant Group. It provides a flexible and efficient way to store, process, and query large amounts of graph-structured data.

IVF_FLAT stands for "Inverted File Flat", which is a storage format used in some graph databases to optimize the storage and querying of graph data. In the context of Milvus, IVF_FLAT might be used as a storage index or compression method to improve query performance and reduce storage costs.

If you have any more specific information about IVF_FLAT in Milvus, I may be able to provide a more detailed explanation.
Based on the available information, here are some common index building parameters for IVF_FLAT in Milvus, along with their short descriptions and default values:

1. **index_type**: The type of index to build (e.g., FLAT, INVERTED).
	* Default: flat
2. **ivf_type**: The type of Inverted File Format to use.
	* Default: FST (Forward Sorted Tree)
3. **leaf_size**: The size of the leaf nodes in the index.
	* Default: 1024
4. **level_num**: The number of levels in the inverted file.
	* Default: 5
5. **node_type**: The type of node to use in the index (e.g., hash, btree).
	* Default: hash
6. **max_key_size**: The maximum size of a key in the index.
	* Default: 1024
7. **leaf_threshold**: The minimum number of leaves required to build an index level.
	* Default: 10
8. **block_size**: The size of each block in the index.
	* Default: 2048

Please note that these parameters and their default values may be subject to change, and it's always recommended to consult the Milvus documentation for the most up-to-date information.

If you have any more specific information about IVF_FLAT in Milvus or would like further clarification on these parameters, feel free to ask!
"""
