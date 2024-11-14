from dotenv import load_dotenv
load_dotenv()  # noqa

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os


USER_AGENT = os.getenv('USER_AGENT')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = 'milvus_docs_collection'


# Use the WebBaseLoader to load specified web pages into documents
loader = WebBaseLoader([
    'https://milvus.io/docs/overview.md',
    'https://milvus.io/docs/release_notes.md',
    'https://milvus.io/docs/architecture_overview.md',
    'https://milvus.io/docs/four_layers.md',
    'https://milvus.io/docs/main_components.md',
    'https://milvus.io/docs/data_processing.md',
    'https://milvus.io/docs/bitset.md',
    'https://milvus.io/docs/boolean.md',
    'https://milvus.io/docs/consistency.md',
    'https://milvus.io/docs/coordinator_ha.md',
    'https://milvus.io/docs/replica.md',
    'https://milvus.io/docs/knowhere.md',
    'https://milvus.io/docs/schema.md',
    'https://milvus.io/docs/dynamic_schema.md',
    'https://milvus.io/docs/json_data_type.md',
    'https://milvus.io/docs/metric.md',
    'https://milvus.io/docs/partition_key.md',
    'https://milvus.io/docs/multi_tenancy.md',
    'https://milvus.io/docs/timestamp.md',
    'https://milvus.io/docs/users_and_roles.md',
    'https://milvus.io/docs/index.md',
    'https://milvus.io/docs/disk_index.md',
    'https://milvus.io/docs/scalar_index.md',
    'https://milvus.io/docs/performance_faq.md',
    'https://milvus.io/docs/product_faq.md',
    'https://milvus.io/docs/operational_faq.md',
    'https://milvus.io/docs/troubleshooting.md',
])

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
connection_args = {"uri": MILVUS_URI}

vector_store = Milvus.from_documents(
    documents=all_splits,
    embedding=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
    drop_old=True,
)

query = "What are the main components of Milvus?"
docs = vector_store.similarity_search(query)

print(len(docs))
for doc in docs:
    print(doc)
